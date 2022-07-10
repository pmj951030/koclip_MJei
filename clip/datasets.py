from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms



from glob import glob

import PIL
import pandas as pd
import clip
from transformers import AutoModel, AutoTokenizer
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageTextPairDataset(Dataset):
    def __init__(self):
        
#         self.image_list = glob("/media/lsh/Samsung_T5/koclip_dataset/*.png")
#         self.image_text_dataframe = pd.read_csv("./korea.csv")
        
        ###  MJei ####
        ## 파일 불러오기
        with open('D:/coco_data/MSCOCO_train_val_Korean.json') as f:
            self.json_data = json.load(f)
        json_dataframe=pd.DataFrame(self.json_data)
        json_dataframe['file_path']=json_dataframe['file_path'].apply(lambda x : 'D:/coco_data/'+x)
        
        
        def remove_text_len_100(self,input_list_text):
            """
            텍스트 길이(띄어쓰기포함)100글자 이상인것은 caption_ko에서 제거함 ,
            데이터 보았을때 kobert토큰 개수 77이상일때 최소 글자길이 103이었기 때문
            """
            list_result=[text for text in input_list_text if len(text)<=100 and text!='']

            return list_result
        
        json_dataframe['caption_ko']=json_dataframe['caption_ko'].apply(lambda x : remove_text_len_100(x))
        
        json_train_data=json_dataframe[json_dataframe['file_path'].str.contains('train')] ## train만 추출
        
        ## 한문장으로 만들음 -> 분리시키려고
        json_train_data['caption_ko']=json_train_data['caption_ko'].apply(lambda x : '!@#'.join(x)) 

        
        json_train_data=json_train_data.head(100) ## 데이터 일부만 작업
        
        ## 리스트 문장 연결시킨거 다시 구분자로 분리
        train_data=json_train_data['caption_ko'].str.split('!@#') ## 콤마단위로 분리
        train_data=train_data.apply(lambda x : pd.Series(x)) ## 시리즈로 변경
        train_data = train_data.stack().reset_index(level=1, drop=True).to_frame('caption_ko') ## 분리된 데이터 생성
        json_train_data.drop('caption_ko',axis=1,inplace=True) ## 중복되는 열 제거
        train_data = json_train_data.merge(train_data, left_index=True, right_index=True, how='left') ## 연결
        train_data.reset_index(drop=True,inplace=True)
        
        self.image_list = list(train_data['file_path'].values)
        self.image_text_dataframe = train_data
        
        
        ##### MJei #####
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)
        
        
        

        
        
    def __getitem__(self, idx):

        
        
        try:
            image_path = self.image_list[idx]

            text_idx = idx##int(image_path.split("/")[-1].split(".")[0])
            text_prompt = self.image_text_dataframe.iloc[text_idx]["caption_ko"] ## text
        
            image = self.preprocess(PIL.Image.open(image_path))
            image_tensor = image

            text_tensor = self.tokenizer(
                text_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                add_special_tokens=True,
                return_token_type_ids=False
            )


            input_ids = text_tensor['input_ids'][0]
            attention_mask = text_tensor['attention_mask'][0]
            return image_tensor, input_ids, attention_mask # tensor : 3 x 224 x 224, "happy cat"
        
        except:
            return self.__getitem__(idx + 1)
            
        

    def __len__(self):
        
        return len(self.image_list)



# dataset = ImageTextPairDataset()