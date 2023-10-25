# coding: UTF-8
"""Load multimodal data with arbitrary ids for visualization"""
from importlib import import_module
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from utils import cut_all_texts, set_seed, build_vocab
from pytorch_grad_cam import GradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


with open("Weibo_anxiety/data/cn_stopwords.txt", encoding='utf-8') as f:
    stopwords = f.read()
stopwords_list = stopwords.split('\n')
stopwords_list.append('焦虑症')
stopwords_list.append('\n')


def get_feature_and_label(id, config, vocab, pad_size=32):
    def pad(x):
        return x[:pad_size] if len(x) > pad_size else x + [0] * (pad_size - len(x))

    df = pd.read_excel(config.train_path)
    row = df[df['id'] == id]
    cut_texts = cut_all_texts(row['content'])
    nlp_feature = torch.tensor(
        [pad([vocab[word] for word in words if word not in stopwords_list]) for words in cut_texts])
    trans = transforms.ToTensor()
    image_path = config.image_path + '/(' + str(id) + ')/1.jpg'
    img = Image.open(image_path)
    image = img.convert('RGB')
    image = image.resize((224, 224))
    image = trans(image)
    cv_feature = torch.unsqueeze(image, 0)
    behavior_data = row.iloc[:, [2, 4, 5, 6]]
    behavior_feature = torch.tensor(behavior_data.to_numpy())
    feature = torch.cat((nlp_feature, cv_feature.reshape(cv_feature.shape[0], -1), behavior_feature), 1)
    label = torch.tensor(row['total'].reset_index(drop=True))
    return nlp_feature, image, feature, label, img


def get_grad_cam(model, target_layers, image, feature, out, nlp_feature, img):
    cam = EigenGradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=feature)

    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = torch.Tensor(grayscale_cam)
    grayscale_cam = grayscale_cam.narrow(1, nlp_feature.shape[1], 224*224*3)
    grayscale_cam = grayscale_cam.reshape(3, 224, 224)
    grayscale_cam = grayscale_cam.sum(axis=0)
    visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam.numpy(), use_rgb=True)
    out = out.squeeze(0).cpu().detach().numpy()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.suptitle("Predict:[{:.2f},{:.2f}] , Label:1 , State:Anxiety, ID:51".format(out[0], out[1]), x=0.5, y=0.85)
    plt.subplot(121), plt.title("Raw"), plt.axis('off'),
    plt.imshow(img.resize((224, 224)))
    plt.subplot(122), plt.title("Eigen-grad-CAM"), plt.axis('off'),
    plt.imshow(visualization)
    # plt.savefig("figure1028.png")
    plt.show()


if __name__ == '__main__':
    id = 51
    set_seed()
    dataset = 'Weibo_anxiety'
    embedding = 'random'
    model_name = 'MADNet'
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    config.batch_size = 1
    data = pd.read_csv("Weibo_anxiety/data/final_data_11_13.csv")
    vocab, stopword_list, avg_len = build_vocab(data, min_freq=config.min_freq)
    config.n_vocab = len(vocab)

    # loading model
    model = x.Model(config)
    checkpoint = torch.load(config.f1_save_path)
    model.load_state_dict(checkpoint)
    model.to(config.device).eval()

    # visualization
    nlp_feature, image, feature, label, img = get_feature_and_label(id, config, vocab)
    out = model(feature)
    target_layers = [model.cv.conv]
    get_grad_cam(model, target_layers, image, feature, out, nlp_feature, img)
