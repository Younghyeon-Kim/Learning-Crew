
# 필요한 모듈 임포트
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from tensorflow.keras.optimizers import Adam


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
#토크나이징, 임베딩 진행



#################
##CNN
#################

import itertools
voc = set(list(itertools.chain.from_iterable(tk_q)))
label = data_frame['Intent']
word_len = [len(data_frame['Q'][i]) for i in data_frame.index] 


# 학습용, 검증용, 테스트용 데이터셋 생성 ○3
# 학습셋:검증셋:테스트셋 = 7:2:1
ds = tf.data.Dataset.from_tensor_slices((q_train, label))
ds = ds.shuffle(len(q_train))

train_size = int(len(q_train) * 0.7)
val_size = int(len(q_train) * 0.2)
test_size = int(len(q_train) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)






# 하이퍼 파라미터 설정

VOCAB_SIZE = len(voc) # 사전 갯수
intent_num = len(data_frame['Intent'].unique())     #라벨 갯수(예: 제품, 주문, 행사 등등)
MAX_SEQ_LEN = int(sum(word_len)/len(word_len))  #문장 당 평균 단어 갯수


dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5


# CNN 모델 정의  ○4
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3,4,5gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(intent_num, name='logits')(dropout_hidden)
predictions = Dense(intent_num, activation=tf.nn.softmax)(logits)


# 모델 생성  ○5
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()




# 모델 학습 ○6
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가(테스트 데이터 셋 이용) ○7
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
print('loss: %f' % (loss))

# 모델 저장  ○8
model.save(data_path+ 'intent_model.h5')