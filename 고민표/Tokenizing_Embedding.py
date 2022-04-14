from konlpy.tag import Okt, Mecab
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pickle
from tqdm.auto import tqdm



#############################
### 1. 파일 불러오기
#############################

data_path = '/content/drive/MyDrive/'

train_data = pd.read_csv(data_path + 'cafe_qa_train.csv',index_col = 0)
test_data = pd.read_csv(data_path + 'cafe_qa_validation.csv', index_col = 0)

#웹사이트, AS 없애기 (일단 카페 데이터만 해당)
condition = train_data[(train_data.Intent == '웹사이트') | (train_data.Intent == 'AS')].index
train_data.drop(condition, axis = 0, inplace = True)


data_frame = train_data

#################################
### 2. Intent 항목 레이블, 원핫인코딩(의도분석에 이용)
#################################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#레이블 인코딩
lab_encoder = LabelEncoder()
lab_encoder.fit(data_frame['Intent'].unique())
lst = lab_encoder.classes_
label_list = dict(zip(lst, range(len(lst))))

print(label_list)    #원래값-숫자 목록

#레이블 데이터 적용
lab_in=lab_encoder.transform(data_frame['Intent'])   # label 값
ori_in=lab_encoder.inverse_transform(lab_in)              # 원래 값
data_frame['Intent'] = lab_in

#lable 값 저장(pkl파일로)
with open(data_path + 'cafe_i_train.pkl','wb') as f:
    pickle.dump(data_frame['Intent'],f)  

#pkl파일 불러오기
with open(data_path + 'cafe_i_train.pkl','rb') as f:
     cafe_i_train = pickle.load(f)

#원핫 인코딩(선택)
data_frame = pd.get_dummies(data_frame, columns = ['Intent'])




#################################
### 3. 토크나이징(단어 분리)
#################################

stop_words = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']
mecab = Mecab()
clean_train = []

def preprocessing(review, mecab, remove_stopwords = False, stop_words = []):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]","", review) # 한글을 제외한 문자들 공백으로 치환
    word_review = mecab.morphs(review_text)

    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]

    return word_review

def cleaning(df):
    clean_data = []
    for i, review in tqdm(enumerate(df), total = len(df)):
        # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
        if type(review) == str:
            p = preprocessing(review, mecab, remove_stopwords=True, stop_words=stop_words)
            clean_data.append(p)
        else:
            clean_data.append([]) #string이 아니면 빈 값 추가
            print(i, review)

    return clean_data

def qa_tk(df):
  global data_path
  tk_q = cleaning(df['Q'])
  tk_a = cleaning(df['A'])
  with open(data_path + 'cafe_tk_q_train.pkl','wb') as f:
    pickle.dump(tk_q,f)
  with open(data_path + 'cafe_tk_a_train.pkl','wb') as f:
    pickle.dump(tk_a,f)
  return tk_q, tk_a

#토크나이징 하고 파일 저장하기
tk_q, tk_a = qa_tk(data_frame)
print(tk_q[0])
print(tk_a[0])


#######################################################
### 4. 임베딩(분리한 단어를 분석할 수 있게 벡터로 변환)
#######################################################

import itertools
def emb(data):
 
  word_len = [len(data_frame['Q'][i]) for i in data_frame.index]
  MAX_SEQ_LEN = int(sum(word_len)/len(word_len))            #한 문장당 단어 평균 갯수
  voc = set(list(itertools.chain.from_iterable(data)))  #단어 총 갯수
  
  VOCAB_SIZE=len(voc)
  tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
  tokenizer.fit_on_texts(data)          

  # 각 단어를 사전의 index로 표현
  train_seq = tokenizer.texts_to_sequences(data)
  

  #단어 사전 생성
  word2idx = {k:v for k, v in tokenizer.word_index.items() if v < VOCAB_SIZE}
  word2idx['<PAD>'] = 0
  
  
  
  x_train = pad_sequences(train_seq, maxlen=MAX_SEQ_LEN, padding='pre', truncating='post')     # padding = 'pre' 하면 패딩값(0)이 앞에부터 채워짐.
  
  return x_train

#임베딩 하고 저장하기
emb_q_train = emb(tk_q)
emb_a_train = emb(tk_a)

print(emb_q_train[0])
print(emb_a_train[0])

with open(data_path + 'cafe_emb_q_train.pkl', 'wb') as f:
  pickle.dump(emb_q_train, f)
with open(data_path + 'cafe_emb_a_train.pkl', 'wb') as f:
  pickle.dump(emb_a_train, f)