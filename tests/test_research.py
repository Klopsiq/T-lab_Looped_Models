import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from looped_models.model import LoopedLM, ModelConfig, rotate
from looped_models.experiment import atomically_save, evaluate, train_one


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model=LoopedLM(ModelConfig(width=32,intermediate=64,relative=True,depth_rope=True))
        self.x=torch.randint(256,(2,12))

    def test_causality(self):
        other=self.x.clone(); other[:,7:]=(other[:,7:]+73)%256
        self.assertTrue(torch.equal(self.model(self.x,4)[:,:7], self.model(other,4)[:,:7]))

    def test_rotation_inverse_norm_and_depth_extension(self):
        x=torch.randn(2,3,32)
        for t in [0,1,63,64,129]:
            y=rotate(x,t,1000,16)
            torch.testing.assert_close(rotate(y,t,1000,16,True),x,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(y.square().sum(-1),x.square().sum(-1))
        torch.testing.assert_close(rotate(x,0,1000,16),x)
        self.assertFalse(torch.allclose(rotate(x,64,1000,16),rotate(x,129,1000,16)))

    def test_readout_matches_separate_forward(self):
        self.model.eval()
        _,states=self.model(self.x,8,collect=True)
        for t in [1,2,4,8]:
            torch.testing.assert_close(self.model.readout(states[t]),self.model(self.x,t),atol=1e-6,rtol=1e-6)

    def test_tying_and_parameter_budget(self):
        count=self.model.parameter_count()
        self.model(self.x,1); self.model(self.x,8)
        self.assertEqual(count,self.model.parameter_count())
        self.assertEqual(LoopedLM(ModelConfig()).parameter_count(),90561)
        self.assertNotIn('lm_head.weight',self.model.state_dict())

    def test_mlp_first_changes_order_without_changing_budget(self):
        standard=LoopedLM(ModelConfig(width=32,intermediate=64,mlp_first=False))
        reversed_order=LoopedLM(ModelConfig(width=32,intermediate=64,mlp_first=True))
        reversed_order.load_state_dict(standard.state_dict())
        tokens=torch.tensor([[1,2,3,4]])
        with torch.inference_mode():
            standard_logits=standard(tokens,loops=2)
            reversed_logits=reversed_order(tokens,loops=2)
        self.assertEqual(standard.parameter_count(),reversed_order.parameter_count())
        self.assertEqual(standard_logits.shape,reversed_logits.shape)
        self.assertFalse(torch.equal(standard_logits,reversed_logits))

    def test_zero_branch_preserves_external_skip(self):
        for layer in self.model.core.modules():
            if isinstance(layer,torch.nn.Linear): torch.nn.init.zeros_(layer.weight)
        h=torch.randn(2,12,32)*3
        e=torch.randn_like(h)
        torch.testing.assert_close(self.model.step(h,e,5),h,rtol=0,atol=0)

    def test_gradient_checkpointing_equivalence(self):
        other=copy.deepcopy(self.model)
        self.model(self.x,4).square().mean().backward()
        other(self.x,4,grad_checkpoint=True).square().mean().backward()
        for a,b in zip(self.model.parameters(),other.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)

    def test_save_reload_logits(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'; atomically_save(path,self.model.state_dict())
            other=LoopedLM(self.model.cfg); other.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(self.model(self.x,4),other(self.x,4),rtol=0,atol=0)

    def test_small_batch_learns(self):
        seq=torch.arange(13).repeat(2,1)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=0.005)
        values=[]
        for _ in range(30):
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(self.model(seq[:,:-1],2).reshape(-1,256),seq[:,1:].reshape(-1))
            values.append(loss.item()); loss.backward(); optimizer.step()
        self.assertLess(values[-1],values[0]*0.8)

    def test_eval_shift_and_token_aggregation(self):
        class Uniform:
            cfg=ModelConfig()
            alpha=torch.tensor(1.)
            def eval(self): pass
            def __call__(self,x,loops,collect):
                states=[torch.ones(*x.shape,2) for _ in range(loops+1)]
                return self.readout(states[-1]),states
            def readout(self,h): return torch.zeros(*h.shape[:2],256)
        data=np.arange(27,dtype=np.uint8).reshape(3,9)
        result=evaluate(Uniform(),data,np.array(['a','b','b']),[1,2],batch_size=2)
        self.assertEqual(result['tokens'],24)
        self.assertAlmostEqual(result['metrics']['1']['nll'],np.log(256),places=6)
        self.assertEqual(result['per_document'][1]['b']['tokens'],16)

    def test_interrupted_trainer_resumes_exactly(self):
        cfg=dict(width=32,intermediate=64,heads=4,kv_heads=2,core_layers=2,device='cpu',lr=.001,weight_decay=.1,
                 train_loops=2,steps=4,batch_size=2,context=8,warmup_steps=1,eval_every=2,grad_checkpoint=False,eval_loops=[1,2])
        train=np.random.default_rng(2).integers(0,256,(8,9),dtype=np.uint8)
        ids=np.array(['a']*8)
        with tempfile.TemporaryDirectory() as d:
            full=dict(cfg,output=str(Path(d)/'full')); interrupted=dict(cfg,output=str(Path(d)/'interrupted'))
            Path(full['output']).mkdir(); Path(interrupted['output']).mkdir()
            train_one(full,'relative',17,train,train,ids,{'hashes':{}})
            def save_and_interrupt(path,obj):
                atomically_save(path,obj)
                raise InterruptedError('Deliberate interruption after checkpoint')
            with patch('looped_models.experiment.atomically_save',side_effect=save_and_interrupt):
                with self.assertRaises(InterruptedError):
                    train_one(interrupted,'relative',17,train,train,ids,{'hashes':{}})
            train_one(interrupted,'relative',17,train,train,ids,{'hashes':{}})
            a=torch.load(Path(full['output'])/'relative_s17/last.pt',weights_only=False)
            b=torch.load(Path(interrupted['output'])/'relative_s17/last.pt',weights_only=False)
            self.assertEqual(a['tokens_seen'],b['tokens_seen'])
            for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            # Resume after the final checkpoint but before result.json was written.
            (Path(interrupted['output'])/'relative_s17/result.json').unlink()
            train_one(interrupted,'relative',17,train,train,ids,{'hashes':{}})


if __name__=='__main__': unittest.main()
