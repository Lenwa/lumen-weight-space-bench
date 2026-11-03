import unittest
import torch
from edit_logic import normalized_step, temporary_weight, grade, comparison, local_gradient, is_resource_failure, failure_receipt

class EditTests(unittest.TestCase):
    def test_resource_failures_cannot_become_resident_errors(self):
        self.assertTrue(is_resource_failure(MemoryError('RAM floor')))
        self.assertTrue(is_resource_failure(torch.cuda.OutOfMemoryError('CUDA OOM')))
        self.assertTrue(is_resource_failure(RuntimeError('CUDA out of memory')))
        self.assertFalse(is_resource_failure(ValueError('no-op outputs differ')))

    def test_failed_generation_retains_payload_and_active_case(self):
        error=ValueError('empty final');error.result=dict(output_ids=[1,2],generation={'max_new_tokens':128})
        r=failure_receipt(error,dict(phase='generation',arm='edited',case='h1'))
        self.assertEqual(r['native_result'],error.result)
        self.assertEqual(r['active']['case'],'h1')

    def test_inference_capture_can_feed_cpu_gradient(self):
        with torch.inference_mode():
            x=torch.randn(2,3);z=torch.randn(2,4)
        g,_=local_gradient(x,z,torch.tensor([1,2]),torch.ones(4),torch.randn(5,4),1e-6)
        self.assertEqual(tuple(g.shape),(4,3))
        self.assertTrue(torch.isfinite(g).all())

    def test_local_gradient_matches_full_small_matrix_autograd(self):
        torch.manual_seed(42)
        x=torch.randn(3,5);base=torch.randn(3,4)
        w=torch.randn(4,5,requires_grad=True);norm=torch.randn(4);head=torch.randn(7,4)
        labels=torch.tensor([1,3,4]);z=base+x@w.T
        logits=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*norm)@head.T
        expected,=torch.autograd.grad(torch.nn.functional.cross_entropy(logits,labels),w)
        actual,_=local_gradient(x,z.detach(),labels,norm,head,1e-6)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=1e-5))

    def test_gradient_direction_and_norm(self):
        w=torch.tensor([[1.,2.],[3.,4.]])
        g=torch.tensor([[2.,-1.],[1.,3.]])
        candidate=normalized_step(w,g,.001)
        d=candidate-w
        self.assertLess(float((d*g).sum()),0)
        self.assertAlmostEqual(float(d.norm()/w.norm()),.001,places=7)

    def test_bad_gradient_refused(self):
        for g in [torch.zeros(2,2),torch.full((2,2),float('nan'))]:
            with self.assertRaises(ValueError):normalized_step(torch.ones(2,2),g,.001)

    def test_real_module_rollback_on_exception(self):
        m=torch.nn.Linear(2,2,bias=False)
        w=m.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with temporary_weight(m,w+1):
                self.assertTrue(torch.equal(m.weight,w+1))
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(m.weight,w))

    def test_grade_rejects_footer_boolean_and_truncation(self):
        self.assertTrue(grade('{"count":17}',True,17))
        for text,complete in [('{"count":17}\n—Assistant',True),('{"count":true}',True),('{"count":17}',False),('{"count":17,"extra":0}',True)]:
            self.assertFalse(grade(text,complete,17))

    def test_comparison_requires_every_arm_case_and_exact_noop(self):
        cases=[dict(id='h',split='heldout',expected=17),dict(id='r',split='retention',expected=2)]
        rows=[dict(arm=a,case=c['id'],answer='{"count":%d}'%c['expected'],complete=True,output_ids=[1,2]) for a in ['original','noop','edited','rollback'] for c in cases]
        self.assertTrue(comparison(cases,rows)['complete'])
        self.assertFalse(comparison(cases,rows[:-1])['complete'])
        rows[2]['output_ids']=[3]
        self.assertFalse(comparison(cases,rows)['noop_exact'])

if __name__=='__main__':unittest.main()
