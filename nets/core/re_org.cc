#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ReOrg")
    .Input("to_reorg: float")
    .Output("reorged: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });



class ReOrgOp : public OpKernel {
 public:
  explicit ReOrgOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    const TensorShape & input_tensor_shape = input_tensor.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_shape,
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    //for (int i = 1; i < N; i++) {
      //output(i) = 0;
    //}
    int stride =2;

    int batch = input_tensor_shape.dim_size(0);
    int h = input_tensor_shape.dim_size(1);
    int w = input_tensor_shape.dim_size(2);
    int c = input_tensor_shape.dim_size(3);
    int b,i,j,k;
    int out_c = c/(stride*stride);

    int forward =0;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) output(out_index) = input(in_index);
                    else output(in_index) =input(out_index);
                }
            }
        }
    }


    // Preserve the first input value if possible.
    //if (N > 0) output(0) = input(0);
  }
};
REGISTER_KERNEL_BUILDER(Name("ReOrg").Device(DEVICE_CPU), ReOrgOp);
