# -*- coding: utf-8 -*-
"""chatbot_data_add.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SMfNYtDBFc-QQUU0-dvyl19kLoIuOygo
"""

import pandas as pd
import os

path = "/content/drive/MyDrive/BIGDATA_STUDY/project_NLP/"
#파일 불러오기
df1 = pd.read_csv(path + "카페_validation.csv")
df2 = pd.read_csv(path + "카페_train.csv")
os.getcwd()

#c-a , s-q 지우기
no_data1 = df1[ (df1['발화자'] == 's') & (df1['QA여부'] == 'q')].index
no_data2 = df1[ (df1['발화자'] == 'c') & (df1['QA여부'] == 'a')].index
df1_dr = df1.drop(no_data1)


no_data3 = df2[ (df2['발화자'] == 's') & (df2['QA여부'] == 'q')].index
no_data4 = df2[ (df2['발화자'] == 'c') & (df2['QA여부'] == 'a')].index
df2_dr = df2.drop(no_data3)

df1_dr.drop(no_data2, inplace = True)
df1_dr.reset_index(inplace = True, drop = True)

df2_dr.drop(no_data4, inplace = True)
df2_dr.reset_index(inplace = True, drop = True)

# qa 정렬대로 뽑아오기 
Q = []
A = []
Intent = []

for i in df1_dr.index:
  if df1_dr['QA여부'][i] == 'q' and df1_dr['QA여부'][i+1] == 'a':
    Q.append(df1_dr.loc[i]['발화문'])
    A.append(df1_dr.loc[i+1]['발화문'])
    Intent.append(df1_dr.loc[i]['인텐트'])

cafe_test_data= pd.DataFrame({'Q':Q,
                'A':A,
                'label':Intent})

# qa 정렬대로 뽑아오기
Q = []
A = []
Intent = []

for i in df2_dr.index:
  if df2_dr['QA여부'][i] == 'q' and df2_dr['QA여부'][i+1] == 'a':
    Q.append(df2_dr.loc[i]['발화문'])
    A.append(df2_dr.loc[i+1]['발화문'])
    Intent.append(df2_dr.loc[i]['인텐트'])

data= pd.DataFrame({'Q':Q,
                'A':A,
                'label':Intent})

# 중복 질문 제거
# 데이터: 72559 --> 중복 제거 후 데이터: 66041 
data.drop_duplicates(['Q'], inplace = True)
data.reset_index(inplace = True, drop = True)

# -------------------------------------------
# label 삭제
# 우리 카페에 필요한 데이터만 남기기
# -------------------------------------------

# 매장, 멤버십, 예약, 웹사이트, AS, 제품_추천, 제품_용도, 비교, 재고, 교환, 제품_가격 삭제
del_label = data[data['label'].str.contains('매장|멤버십|예약|웹사이트|AS|제품_추천|제품_용도|비교|재고|교환|제품_가격')].index
data = data.drop(del_label)

data.reset_index(inplace = True, drop = True)

# 필요 없는 데이터 제거
extra_q = data[data['Q'].str.contains('설빙|베스킨라빈스|스타벅스|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|텀블러')].index
data = data.drop(extra_q)

extra_a = data[data['A'].str.contains('설빙|베스킨라빈스|스타벅스|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|텀블러')].index
data = data.drop(extra_a)

data.reset_index(inplace=True, drop=True)

# 날짜 묻는 질문/날짜로 대답하는 답변 제거
product_date_q = data[data['Q'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_q)

product_date_a = data[data['A'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_a)

data.reset_index(inplace=True, drop=True)


# 배달/배송 관련 정보 제거
delivery_q = data[data['Q'].str.contains('배달|택배')].index
data = data.drop(delivery_q)

delivery_a = data[data['A'].str.contains('배달|택배')].index
data = data.drop(delivery_a)

delivery = data[data['label'].str.contains('배송')].index
data = data.drop(delivery)

data.reset_index(inplace=True, drop=True)

# 수거 제거
waste_q = data[data['Q'].str.contains('수거')].index
data = data.drop(waste_q)

waste_a = data[data['A'].str.contains('수거')].index
data = data.drop(waste_a)

data.reset_index(inplace=True, drop=True)

# 총 얼마예요? 제거
total_q = data[data['Q'].str.contains('총 얼마예요?')].index
data = data.drop(total_q)

data.reset_index(inplace=True, drop=True)

# 품절 제거
absence_q = data[data['Q'].str.contains('품절')].index
data = data.drop(absence_q)

absence_a = data[data['A'].str.contains('품절')].index
data = data.drop(absence_a)

data.reset_index(inplace=True, drop=True)

# 원두 삭제
coffee_q = data[data['Q'].str.contains('원두')].index
data = data.drop(coffee_q)

coffee_a = data[data['A'].str.contains('원두')].index
data = data.drop(coffee_a)

data.reset_index(inplace=True, drop=True)

# 재료 변경 불가 안내 문구 학습
data.loc[data['Q'].str.contains('빼|교환|바꾸|추가|바꿔|뺄'), 'A'] = '현재 재료 변경은 불가능합니다.'

# 디카페인 불가 안내 문구 학습
data.loc[data['Q'].str.contains('디카페인|디 카페인'), 'A'] = '현재 디카페인 음료는 제공되지 않습니다.'

# 제품 불량 문의 안내 문구 학습
data.loc[data['label'].str.contains('제품_불량'), 'A'] = '죄송합니다. 카운터에서 확인 도와드리겠습니다.'

# 분할 결제 불가 안내 문구 학습
data.loc[data['Q'].str.contains('분할 결제|분할결제|나눠서 결제'), 'A'] = '현재 분할 결제가 불가능합니다. 다른 결제 수단을 선택해주세요.'

# 할인 불가 안내 문구 학습
data.loc[data['Q'].str.contains('할인|깎|깍'), 'A'] = '현재 할인은 불가합니다.'

# 칼로리 정보 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('열량|칼로리'), 'A'] = '현재 칼로리 정보는 제공되지 않습니다. 카운터에 문의 바랍니다.'

# 행사 안내 문구 학습
data.loc[data['Q'].str.contains('행사|이벤트'), 'A'] = '현재 진행중인 행사가 없습니다. 다음 행사를 기다려주세요~^^'

# lst = list(cafe_train_data['Intent'])
# lst2 = [i.split('_')[0] if i.split('_')[0] != '제품' else i.split('_')[0] +'_'+ i.split('_')[2] for i in lst]Q

data

"""#추가 데이터 병합"""

path = "/content/drive/MyDrive/BIGDATA_STUDY/project_NLP/"
#QA추가 csv 파일 불러오기
df_add = pd.read_csv(path + "cafe_add.csv")
os.getcwd()

df_add

df_comb = pd.concat([data, df_add])

df_comb

#################################
### label 항목 레이블, 원핫인코딩(의도분석에 이용)
#################################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#레이블 인코딩
lab_encoder = LabelEncoder()
lab_encoder.fit(df_comb['label'].unique())
lst = lab_encoder.classes_
label_list = dict(zip(lst, range(len(lst))))

print(label_list)    #원래값-숫자 목록

#레이블 데이터 적용
lab_in=lab_encoder.transform(df_comb['label'])   # label 값
ori_in=lab_encoder.inverse_transform(lab_in)         # 원래 값
df_comb['label'] = lab_in

df_comb

"""#데이터 저장"""

df_comb.to_csv(path + 'ChatbotData.csv')

pd.read_csv(path +'ChatbotData.csv')

cafe_test_data.to_csv(path + 'cafe_qa_validation.csv')

pd.read_csv(path +'cafe_qa_validation.csv')