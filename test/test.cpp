#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <Windows.h>

//struct Net : torch::nn::Module
//{
//    Net(int64_t n, int64_t m)
//    {
//        w = register_parameter("w", torch::randn({n,m}));
//        b = register_parameter("b", torch::randn(m));
//    }
//    torch::Tensor forward(torch::Tensor input)
//    {
//        return torch::addmm(b, input, w);
//    }
//    torch::Tensor w,b;
//};
//
//struct Net : torch::nn::Module 
//{
//	Net(int64_t N, int64_t M)
//		: linear(register_module("linear", torch::nn::Linear(N, M))) 
//	{
//		another_bias = register_parameter("b", torch::randn(M));
//	}
//	
//	torch::Tensor forward(torch::Tensor input) 
//	{
//		return linear(input) + another_bias;
//	}
//	torch::nn::Linear linear;
//	torch::Tensor another_bias;
//};


struct DCGANGeneratorImpl : torch::nn::Module
{
	torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
	torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
	DCGANGeneratorImpl(int kNoiseSize) :
		conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
		batch_norm1(256),
		conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
		batch_norm2(128),
		conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
		batch_norm3(64),
		conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("batch_norm1", batch_norm1);
		register_module("batch_norm2", batch_norm2);
		register_module("batch_norm3", batch_norm3);
	}
	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(batch_norm1(conv1(x)));
		x = torch::relu(batch_norm2(conv2(x)));
		x = torch::relu(batch_norm3(conv3(x)));
		x = torch::tanh(conv4(x));
		return x;
	}
};
TORCH_MODULE(DCGANGenerator);

const int64_t kNoiseSize = 100;
const int64_t kBatchSize = 64;
const int64_t kNumberOfEpochs = 30;


int main() 
{
	DCGANGenerator generator(kNoiseSize);

	torch::nn::Sequential discriminator(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(128),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),

		torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
		torch::nn::Sigmoid());

	std::string path = CMAKE_SOURCE_DIR;
	path += "/data";

	auto dataset = torch::data::datasets::MNIST(path)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader(
		std::move(dataset), 
		torch::data::DataLoaderOptions().batch_size(kBatchSize));

	torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
	torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));

	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
	{
		int64_t batch_index = 0;
		for (torch::data::Example<>&batch : *data_loader)
		{
			discriminator->zero_grad();
			torch::Tensor real_images = batch.data;
			torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
			torch::Tensor real_output = discriminator->forward(real_images);
			torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
			d_loss_real.backward();

			torch::Tensor noise = torch::randn({ batch.data.size(0),kNoiseSize,1,1 });
			torch::Tensor fake_images = generator->forward(noise);
			torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
			torch::Tensor fake_output = discriminator->forward(fake_images.detach());
			torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
			d_loss_fake.backward();

			torch::Tensor d_loss = d_loss_real + d_loss_fake;

			discriminator_optimizer.step();

			generator->zero_grad();
			fake_labels.fill_(1);
			fake_output = discriminator->forward(fake_images);
			torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
			g_loss.backward();
			generator_optimizer.step();

			std::cout << "epoch: " << epoch << "/" << kNumberOfEpochs << ", batch: " << ++batch_index << ", loss: " << d_loss.item<float>() << " " << g_loss.item<float>() << std::endl;
		}

	}
	//for (torch::data::Example<>& batch : *data_loader)
	//{
	//	std::cout << "Batch size: " << batch.data.size(0) << " | Labels: \n";
	//	for(int64_t i = 0; i < batch.data.size(0); ++i)
	//	{
	//		std::cout << batch.target[i].item<int64_t>() << " ";
	//	}
	//	std::cout << std::endl;
	//}

	//Net net(4,5);
	//std::cout << net.forward(torch::ones({ 2,4 })) << std::endl;
	//for(auto& kv: net.named_parameters())
	//{
	//    std::cout << kv.key() << std::endl << kv.value() << std::endl;
	//}
	//for(auto& tensor: net.parameters())
    //{
    //    std::cout << tensor;
    //}
    //torch::Tensor tensor = torch::eye(3);
    //std::cout << tensor << std::endl;
}