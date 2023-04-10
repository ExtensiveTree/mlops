import io
import streamlit as st
from PIL import Image
from torchvision.models import resnet101, ResNet101_Weights


@st.cache_data()
def load_model():
    return resnet101(weights=ResNet101_Weights.DEFAULT, progress=False).eval()


def preprocess_image(img):
    preprocess_img = ResNet101_Weights.DEFAULT.transforms()
    x = preprocess_img(img).unsqueeze(dim=0)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    class_id = preds.argmax().item()
    score = preds[class_id].item()
    category_name = ResNet101_Weights.DEFAULT.meta["categories"][class_id]
    st.write(f"{category_name}: {100 * score:.1f}%")


model = load_model()

st.title('Классификация изображений')
st.markdown('**Для распознавания изображений используется нейронная сеть ResNet101**')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model(x).squeeze(0).softmax(0)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)