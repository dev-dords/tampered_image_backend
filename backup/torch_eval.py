# # Define transformation (same as training)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images
#     transforms.ToTensor(),          # Convert to PyTorch Tensor
#     transforms.Normalize((0.5,), (0.5,))  # Normalize if needed
# ])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load the PyTorch model
# torch1_path = "./model/resnet18_2_ela_model.pth"
# torch1 = torch.load(torch1_path, weights_only = False, map_location=torch.device('cpu'))

# torch2_path = "./model/Resnet18_30ep_model.pth"
# torch2 = torch.load(torch2_path, weights_only = False, map_location=torch.device('cpu'))

# torch3_path = "./model/Resnet50_model.pth"
# torch3 = torch.load(torch3_path, weights_only = False, map_location=torch.device('cpu'))
# torch_models = [torch1]

        # for torch_model in torch_models:
        #     img = image.load_img(ela_path)
        #     img_tensor = transform(img).unsqueeze(0)
        #     torch_model.eval()
        #     with torch.no_grad():
        #         output = torch_model(img_tensor.to(device))  # Predict
        #         prob = torch.softmax(output, dim=1)  # Convert to probabilities
        #         pred_class = torch.argmax(prob, dim=1).item()  # Get the predicted class
        #         torch_pred = prob[0][1].item()*100
        #         print('PyTorch Prediction:', torch_pred)
        #         predictions.append(torch_pred)