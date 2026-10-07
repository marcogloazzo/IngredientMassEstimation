"""V6 synthetic-only checks. No Nutrition5k training, CLIP downloads or depth runs."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"tests"))
from test_v3 import fake_backbone, vocabulary, record, SyntheticPairs
from ingredient_mass.clip_queries import read_clip_queries,encode_text_queries
from ingredient_mass.depth import image_size,validate_depth_config
from ingredient_mass.gate_diagnostics import diagnose_gate
from ingredient_mass.image_training import run_epoch,save_predictions
from ingredient_mass.mass_filtering import apply_mass_filter,calibrate_mass_threshold,v6_metric_details
from ingredient_mass.metrics import full_metrics
from ingredient_mass.model import load_checkpoint
from ingredient_mass.robust_model import class_mass_statistics,robust_mass_loss
from ingredient_mass.spatial_model import MODEL_TYPE_V6,create_image_model,model_from_checkpoint,protected_result_namespaces

CONFIG = json.loads((ROOT/"config/query2label_100_v6.json").read_text(encoding="utf-8"))
SMALL = {**CONFIG,"attention_dim":16,"attention_feedforward":32,"attention_dropout":0.,"dropout":0.,"amp":False,"mass_scale":200.}


def model():
    with patch("torchvision.models.convnext_small",side_effect=fake_backbone):
        return create_image_model(3,SMALL,pretrained=False)


def initialize_clip(m):
    embeddings = np.zeros((3,512),dtype=np.float32)
    embeddings[0,0] = embeddings[1,1] = 1.
    m.heads.initialize_clip_queries(embeddings,np.array([True,True,False]))


def filter_fixture(raw=None):
    p = np.array([[.9,.1],[.1,.9]],dtype=np.float32)
    raw = np.array([[100.,10.],[10.,100.]],dtype=np.float32) if raw is None else raw
    truth = np.array([[100.,0.],[0.,100.]],dtype=np.float32)
    predictions = {"presence":p,"unfiltered_ingredient_masses":raw,"conditional_grams":raw/p}
    targets = {"presence":truth>0,"masses":truth,"total_mass":truth.sum(1),"shares":truth/100,
               "nutrients":np.tile([200.,10.,20.,30.],(2,1)),"dish_ids":np.array(["a","b"])}
    matrix = np.tile([2.,.1,.2,.3],(2,1)).astype(np.float32)
    return predictions,targets,matrix


class V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_train_only_positive_class_quantiles_and_fallback(self):
        cfg = {**SMALL,"conditional_scale":{**SMALL["conditional_scale"],"min_support":2}}
        records = [record("a","train",{"a":10}),record("b","train",{"a":20,"b":100}),
                   record("v","validation",{"a":100000,"b":100000})]
        stats = class_mass_statistics(records,vocabulary(),cfg)
        np.testing.assert_allclose(stats["class_gram_scales"],[19.,84.,84.],rtol=1e-5)
        self.assertEqual(stats["class_support"].tolist(),[2,1,0])
        self.assertEqual(stats["scale_details"][1]["source"],"pooled_train_fallback")

    def test_asl_bias_and_gate_have_no_bce_correction(self):
        records = [record(str(i),"train",{"a":100.} if i==0 else {"b":50.}) for i in range(10)]
        m = model(); m.initialize_from_train(records,vocabulary())
        self.assertFalse(hasattr(m.heads,"positive_weights"))
        self.assertTrue(torch.allclose(m.heads.presence_head.bias.sigmoid(),torch.tensor([.1,.9,.001]),atol=1e-6))
        out = m(torch.rand(1,3,64,64),depth=torch.rand(1,1,64,64))
        self.assertTrue(torch.allclose(out["presence_probabilities"],out["presence_logits"].sigmoid()))
        expected = torch.nn.functional.softplus(m.heads.conditional_head.bias)*m.heads.class_gram_scales
        self.assertTrue(torch.allclose(expected[:2],torch.tensor([100.,50.]),atol=1e-4))

    def test_500_tokens_and_spatial_maps_initialized_to_point_one(self):
        m = model().eval()
        with torch.no_grad(): out = m(torch.rand(1,3,320,320),depth=torch.rand(1,1,320,320),return_attention=True)
        self.assertEqual(out["attention_tokens"].shape,(1,3,500))
        self.assertEqual(out["depth_spatial_gate_stage3"].shape,(1,1,20,20))
        self.assertEqual(out["depth_spatial_gate_stage4"].shape,(1,1,10,10))
        self.assertTrue(torch.allclose(out["depth_spatial_gate_stage3"],torch.full((1,1,20,20),.1),atol=1e-6))
        self.assertFalse(hasattr(m.heads,"depth_gate_logits"))
        with torch.no_grad():
            m.heads.spatial_gates[0][-1].weight.normal_(std=.1)
            changed = m(torch.rand(1,3,320,320),depth=torch.rand(1,1,320,320),return_attention=True)
        self.assertGreater(float(changed["depth_spatial_gate_stage3"].std()),1e-5)

    def test_only_stage_four_and_heads_receive_gradients(self):
        m = model(); m.set_encoder_trainable(True)
        out = m(torch.rand(2,3,64,64),depth=torch.rand(2,1,64,64))
        loss,parts = robust_mass_loss(out,torch.tensor([[1.,0.,0.],[0.,1.,0.]]),
            torch.tensor([[100.,0.,0.],[0.,50.,0.]]),torch.tensor([100.,50.]),torch.ones(2,4),
            torch.ones(3,4),torch.ones(3),SMALL)
        loss.backward()
        for i in range(6):
            self.assertFalse(m.features[i].training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.features[i].parameters()))
        for i in (6,7): self.assertGreater(float(m.features[i].weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.heads.query_projection.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.heads.spatial_gates[0][-1].bias.grad.abs().sum()),0.)
        self.assertGreater(float(m.heads.auxiliary_mass_head[-1].bias.grad.abs().sum()),0.)
        self.assertTrue(all(np.isfinite(v) for v in parts.values()))
        self.assertAlmostEqual(parts["total"],sum(SMALL["loss_weights"][k]*parts[k] for k in SMALL["loss_weights"]),places=5)

    def test_mass_filter_exact_zeros_and_training_dense(self):
        m = model()
        with torch.no_grad():
            m.heads.presence_head.weight.zero_(); m.heads.presence_head.bias.fill_(torch.logit(torch.tensor(.01)))
        image,depth = torch.rand(1,3,64,64),torch.rand(1,1,64,64)
        training = m(image,depth=depth)
        self.assertTrue((training["ingredient_masses"]>0).all())
        m.eval()
        with torch.no_grad(): deployed = m(image,depth=depth)
        self.assertEqual(float(deployed["ingredient_masses"].sum()),0.)
        self.assertEqual(float(deployed["total_mass"]),0.)
        self.assertEqual(float(deployed["shares"].sum()),0.)
        self.assertTrue(torch.isfinite(deployed["shares"]).all())

    def test_conditional_gradient_survives_zero_presence(self):
        conditional = torch.tensor([[10.,20.,30.]],requires_grad=True)
        logits = torch.full((1,3),-100.,requires_grad=True)
        raw = logits.sigmoid()*conditional
        outputs = {"presence_logits":logits,"conditional_grams":conditional,"class_gram_scales":torch.tensor([20.,100.,300.]),
                   "conditional_class_weights":torch.ones(3),"unfiltered_ingredient_masses":raw,
                   "auxiliary_total_mass":torch.tensor([100.])}
        config = {**SMALL,"loss_weights":{k:float(k=="conditional") for k in SMALL["loss_weights"]}}
        loss,_ = robust_mass_loss(outputs,torch.tensor([[1.,0.,0.]]),torch.tensor([[100.,0.,0.]]),
            torch.tensor([100.]),torch.ones(1,4),torch.ones(3,4),torch.ones(3),config)
        loss.backward()
        self.assertGreater(float(conditional.grad[0,0].abs()),0.)
        self.assertEqual(float(conditional.grad[0,1:].abs().sum()),0.)

    def test_clip_other_stays_random_and_projection_is_trainable(self):
        m = model(); other = m.heads.queries.weight[2].detach().clone()
        initialize_clip(m)
        self.assertTrue(bool(m.heads.clip_initialized))
        self.assertTrue(torch.equal(other,m.heads.queries.weight[2]))
        self.assertAlmostEqual(float(m.heads.queries.weight[0].norm()),np.sqrt(512),places=4)
        self.assertTrue(m.heads.queries.weight.requires_grad)
        self.assertTrue(m.heads.query_projection.weight.requires_grad)

    def test_checkpoint_roundtrip_requires_neither_clip_nor_downloads(self):
        m = model().eval(); initialize_clip(m)
        buffer = io.BytesIO()
        torch.save(dict(model_type=MODEL_TYPE_V6,config=SMALL,vocabulary=vocabulary(),model_state=m.state_dict()),buffer)
        buffer.seek(0)
        with patch("torchvision.models.convnext_small",side_effect=fake_backbone) as builder,patch.dict(sys.modules,{"clip":None}):
            restored = model_from_checkpoint(load_checkpoint(buffer),torch.device("cpu"))
        self.assertIsNone(builder.call_args.kwargs["weights"])
        image,depth = torch.rand(1,3,64,64),torch.rand(1,1,64,64)
        with torch.no_grad(): a,b = m(image,depth=depth),restored(image,depth=depth)
        for key in ("ingredient_masses","conditional_grams","auxiliary_total_mass"):
            self.assertTrue(torch.allclose(a[key],b[key]))

    def test_clip_cache_validation_and_frozen_text_encoder(self):
        class FakeClip(nn.Module):
            def __init__(self): super().__init__(); self.parameter = nn.Parameter(torch.ones(1))
            def encode_text(self,tokens): return torch.eye(512)[:tokens.shape[0]]
        encoder = FakeClip()
        module = types.SimpleNamespace(load=lambda *a,**k:(encoder,None),tokenize=lambda texts,**k:torch.zeros(len(texts),1,dtype=torch.int))
        values,mask,texts = encode_text_queries(vocabulary(),SMALL,module,torch.device("cpu"),Path("unused"))
        self.assertFalse(encoder.parameter.requires_grad)
        self.assertEqual(mask.tolist(),[True,True,False]); self.assertEqual(len(texts),2)
        meta = {"schema_version":1,**{k:SMALL["clip_queries"][k] for k in ("model","repo_revision","template")},"vocabulary_sha256":"abc"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"queries.npz"
            np.savez_compressed(path,text_embeddings=values,initialized_mask=mask,
                vocabulary_keys=np.array(["a","b","__other__"]),metadata=np.asarray(json.dumps(meta)))
            actual = read_clip_queries(path,vocabulary(),SMALL,"abc")
            self.assertEqual(actual[0].shape,(3,512))
            with self.assertRaises(ValueError): read_clip_queries(path,vocabulary(),SMALL,"wrong")
            wrong = copy.deepcopy(vocabulary()); wrong["entries"].reverse()
            with self.assertRaises(ValueError): read_clip_queries(path,wrong,SMALL)

    def test_filter_calibration_validation_only_and_guards(self):
        predictions,targets,matrix = filter_fixture()
        threshold,report = calibrate_mass_threshold(predictions,targets,matrix,SMALL,split="validation")
        self.assertGreater(threshold,.1)
        self.assertEqual(report["selected"]["ingredient_l1_g_per_plate"],0.)
        with self.assertRaisesRegex(ValueError,"validation only"):
            calibrate_mass_threshold(predictions,targets,matrix,SMALL,split="test")
        predictions["unfiltered_ingredient_masses"] = np.array([[60.,40.],[40.,60.]],dtype=np.float32)
        threshold,report = calibrate_mass_threshold(predictions,targets,matrix,SMALL,split="validation")
        self.assertLessEqual(threshold,.1)
        self.assertEqual(report["selected"]["total_mass_mae_g"],0.)

    def test_empty_oracle_composition_report_is_not_falsely_valid(self):
        predictions,targets,matrix = filter_fixture()
        empty = apply_mass_filter(predictions,1.)
        report = v6_metric_details(empty,targets,matrix,{**SMALL,"mass_filter":{**SMALL["mass_filter"],"threshold":1.}})
        self.assertEqual(report["mass_filter"]["empty_predicted_plates"],2)
        self.assertFalse(report["composition_with_true_total_mass"]["available"])

    def test_saved_v6_predictions_and_diagnostic_split_gate_vs_cutoff(self):
        m = model().eval(); initialize_clip(m)
        _,predictions,targets = run_epoch(m,DataLoader(SyntheticPairs(),batch_size=2),torch.ones(3,4),
            torch.ones(3),SMALL,torch.device("cpu"),description="Synthetic V6 validation")
        self.assertEqual(predictions["class_gram_scales"].shape,(3,))
        self.assertEqual(predictions["depth_gate_means"].shape,(3,2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.npz"
            save_predictions(path,predictions,targets,vocabulary())
            with np.load(path,allow_pickle=False) as source:
                report = diagnose_gate(dict(source),{"split":"validation","model_type":MODEL_TYPE_V6,
                    "mass_filter":SMALL["mass_filter"]},vocabulary())
        self.assertIn("unfiltered_mae_g",report["global_present"])
        self.assertIn("filter_effect_mae_g",report["classes_by_l1"][0])
        self.assertTrue(any(r["name"]=="unfiltered" for r in report["scenarios"]))

    def test_cache_geometry_unchanged_and_older_namespaces_protected(self):
        v3 = json.loads((ROOT/"config/query2label_100_v3.json").read_text())
        self.assertEqual(image_size(SMALL),320)
        self.assertEqual(validate_depth_config(SMALL),validate_depth_config(v3))
        self.assertEqual(protected_result_namespaces(MODEL_TYPE_V6),{"v2","v3","v4","v5"})

    def test_notebook_safe_flags_paths_and_no_html(self):
        nb = json.loads((ROOT/"COLAB_QUERY2LABEL_V6.ipynb").read_text(encoding="utf-8"))
        flags = {}; code = ""
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"],[])
            source = "".join(cell["source"]); code += source
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,bool):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id.startswith("RUN_"): flags[target.id]=node.value.value
        self.assertTrue(flags and all(v is False for v in flags.values()))
        self.assertIn('V6 = DRIVE / "v6"',code)
        self.assertIn('BRANCH = "v6-asl-spatial-clip"',code)
        self.assertIn("scripts/prepare_clip_queries.py",code)
        self.assertIn('"--clip-queries", CLIP_QUERIES',code)
        self.assertIn("spec.loader.exec_module(reports)",code)
        self.assertNotIn("display(HTML",code)
        self.assertIn("ALLOW_CPU_TRAINING = False",code)

    def test_evaluation_calibration_saves_mass_cutoff_checkpoint(self):
        import evaluate_image_model as evaluation
        from ingredient_mass.data import nutrient_matrix
        m = model().eval(); initialize_clip(m); vocab = vocabulary()
        p = np.array([[.9,.1,.001],[.1,.9,.001]],dtype=np.float32)
        raw = np.array([[100.,10.,.0001],[10.,100.,.0001]],dtype=np.float32)
        truth = np.array([[100.,0.,0.],[0.,100.,0.]],dtype=np.float32)
        predictions = apply_mass_filter({"presence":p,"unfiltered_ingredient_masses":raw,
            "conditional_grams":raw/p,"auxiliary_total_mass":np.array([100.,100.]),
            "class_gram_scales":np.array([100.,100.,100.]),"depth_gate_means":np.full((2,2),.1)},.05)
        targets = {"presence":(truth>0).astype(np.float32),"masses":truth,"total_mass":truth.sum(1),
                   "shares":truth/100,"nutrients":truth@nutrient_matrix(vocab),"dish_ids":np.array(["a","b"])}
        records = [record("a","train",{"a":100}),record("b","train",{"b":100})]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); checkpoint = base/"best.pt"; output = base/"evaluation"
            torch.save(dict(model_type=MODEL_TYPE_V6,config=SMALL,vocabulary=vocab,model_state=m.state_dict(),epoch=2),checkpoint)
            argv = ["evaluate_image_model.py","--checkpoint",str(checkpoint),"--output-dir",str(output),"--split","validation","--calibrate","--device","cpu"]
            loader = types.SimpleNamespace(dataset=[0,1])
            with patch.object(evaluation,"load_prepared",return_value=(records,vocab)),patch.object(evaluation,"make_loader",return_value=loader),\
                 patch("ingredient_mass.depth.validate_depth_cache"),patch.object(evaluation,"run_epoch",return_value=({"total":0.},predictions,targets)),\
                 patch.object(evaluation,"prepared_fingerprint",return_value={"records.jsonl":"r","vocabulary.json":"v"}),\
                 patch("torchvision.models.convnext_small",side_effect=fake_backbone),patch.object(sys,"argv",argv),patch("sys.stdout",new_callable=io.StringIO):
                evaluation.main()
            result = json.loads((output/"metrics.json").read_text())
            calibrated = load_checkpoint(output/"calibrated.pt")
            self.assertGreater(calibrated["config"]["mass_filter"]["threshold"],.1)
            self.assertTrue(calibrated["calibration"]["changes_ingredient_grams"])
            self.assertEqual(result["end_to_end_lightweight_mass"]["ingredient_mass_l1_per_plate_grams"],0.)
            self.assertGreater(result["unfiltered_end_to_end"]["ingredient_mass_l1_per_plate_grams"],0.)
            with np.load(output/"predictions.npz",allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved["ingredient_masses"],truth))


if __name__ == "__main__": unittest.main()
