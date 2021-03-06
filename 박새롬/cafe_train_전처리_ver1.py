import pandas as pd

cafe_data = pd.read_csv('/content/drive/MyDrive/미니 프로젝트/data/카페_train.csv')

df = cafe_data.dropna(axis=1)
df.head()


# ==========================================

# 직원이 질문하는 경우 & 소비자가 대답하는 경우 삭제

x1 = df[(df['발화자']=='s')&(df['QA여부']=='q')].index
x2 = df[(df['발화자']=='c')&(df['QA여부']=='a')].index

df = df.drop(x1)
df = df.drop(x2)
df.reset_index(inplace = True, drop = True)



# ==========================================
# QA 정렬대로 뽑아오기
Q = []
A = []
Intent = []

for i in df.index:
  if df['QA여부'][i] == 'q' and df['QA여부'][i+1] == 'a':
    Q.append(df.loc[i]['발화문'])
    A.append(df.loc[i+1]['발화문'])
    Intent.append(df.loc[i]['인텐트'])

data= pd.DataFrame({'Q':Q,
                'A':A,
                'Intent' :Intent})


data


# ==========================================
# 중복 질문 제거
# 데이터: 72559 --> 중복 제거 후 데이터: 66041 
data.drop_duplicates(['Q'], inplace = True)
data.reset_index(inplace = True, drop = True)


# ==========================================
# Intent 삭제
# 우리 카페에 필요한 데이터만 남기기
# ==========================================
# 매장, 멤버십, 예약, 행사, 웹사이트, AS, 제품_추천, 제품_용도, 비교, 재고, 교환, 제품_가격 삭제
del_intent = data[data['Intent'].str.contains('매장|멤버십|예약|행사|웹사이트|AS|제품_추천|제품_용도|비교|재고|교환|제품_가격')].index
data = data.drop(del_intent)

data.reset_index(inplace = True, drop = True)

#-------------------------------------------
# 필요 없는 데이터 제거
extra_q = data[data['Q'].str.contains('설빙|베스킨라빈스|스타벅스|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|텀블러')].index
data = data.drop(extra_q)

extra_a = data[data['A'].str.contains('설빙|베스킨라빈스|스타벅스|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|텀블러')].index
data = data.drop(extra_a)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 날짜 묻는 질문/날짜로 대답하는 답변 제거

# -------------------------------------------
# Intent 삭제
# 데이터: 66041 --> Intent 삭제 후 데이터: 50201
# -------------------------------------------

# 매장, 멤버십, 예약, 행사, 웹사이트, AS
del_intent = data[data['Intent'].str.contains('매장|멤버십|예약|행사|웹사이트|AS')].index
data = data.drop(del_intent)

# 제품_불량
product_poor = data[data['Intent'].str.contains('제품_불량')].index
data = data.drop(product_poor)

# 제품_추천
product_recommend = data[data['Intent'].str.contains('제품_추천')].index
data = data.drop(product_recommend)

# 제품_용도
product_usage = data[data['Intent'].str.contains('제품_용도')].index
data = data.drop(product_usage)

data.reset_index(inplace = True, drop = True)

# 날짜 묻는 질문/날짜로 대답하는 답변 제거
# 데이터: 50201 --> 제거 후 데이터: 50034/49582

product_date_q = data[data['Q'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_q)

product_date_a = data[data['A'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_a)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 배달/배송 관련 정보 제거
delivery_q = data[data['Q'].str.contains('배달|택배')].index
data = data.drop(delivery_q)

delivery_a = data[data['A'].str.contains('배달|택배')].index
data = data.drop(delivery_a)

delivery = data[data['Intent'].str.contains('배송')].index
data = data.drop(delivery)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 수거 제거
waste_q = data[data['Q'].str.contains('수거')].index
data = data.drop(waste_q)

waste_a = data[data['A'].str.contains('수거')].index
data = data.drop(waste_a)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 총 얼마예요? 제거
total_q = data[data['Q'].str.contains('총 얼마예요?')].index
data = data.drop(total_q)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 품절 제거
absence_q = data[data['Q'].str.contains('품절')].index
data = data.drop(absence_q)

absence_a = data[data['A'].str.contains('품절')].index
data = data.drop(absence_a)

data.reset_index(inplace=True, drop=True)

#-------------------------------------------
# 원두 삭제
coffee_q = data[data['Q'].str.contains('원두')].index
data = data.drop(coffee_q)

coffee_a = data[data['A'].str.contains('원두')].index
data = data.drop(coffee_a)

data.reset_index(inplace=True, drop=True)


# ==========================================
# 안내 문구 학습
# ==========================================
# 재료 변경 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('빼|교환|바꾸|추가|바꿔|뺄'), 'A'] = '현재 재료 변경은 불가능합니다. 다시 주문해주세요.'

# 디카페인 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('디카페인|디 카페인'), 'A'] = '현재 디카페인 음료는 제공되지 않습니다. 다시 주문해주세요.'

# 제품 불량 문의 안내 문구 학습
data.loc[data['Intent'].str.contains('제품_불량'), 'A'] = '죄송합니다. 카운터에서 확인 도와드리겠습니다.'

# 분할 결제 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('분할 결제|분할결제'), 'A'] = '현재 분할 결제는 불가능합니다. 다른 결제 수단을 선택해주세요.'

# 칼로리 정보 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('열량|칼로리'), 'A'] = '현재 칼로리 정보는 제공되지 않습니다. 카운터에 문의 바랍니다.'


#-------------------------------------------
# 인텐트 제거
data = data.drop(['Intent'], axis=1)


#------------------------------------------- 
# 파일 저장
# DATA_PATH = '/content/drive/MyDrive/미니 프로젝트/data/'
# data.to_csv(DATA_PATH + 'cafe_qa_train_실험용.csv', index=False)


# lst = list(data['Intent'])
# lst2 = [i.split('_')[0] if i.split('_')[0] != '제품' else i.split('_')[0] +'_'+ i.split('_')[2] for i in lst]

# data['Intent'] = lst2


lst = list(data['Intent'])
lst2 = [i.split('_')[0] if i.split('_')[0] != '제품' else i.split('_')[0] +'_'+ i.split('_')[2] for i in lst]

data['Intent'] = lst2

data['Intent'].value_counts()

DATA_PATH = '/content/drive/MyDrive/미니 프로젝트/data/'
data.to_csv(DATA_PATH + 'cafe_qa_train.csv', index=False)


# data_frame = data

# #################################
# ### 2. Intent 항목 레이블, 원핫인코딩(의도분석에 이용)
# #################################

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

# #레이블 인코딩
# lab_encoder = LabelEncoder()
# lab_encoder.fit(data_frame['Intent'].unique())
# lst = lab_encoder.classes_
# label_list = dict(zip(lst, range(len(lst))))

# print(label_list)    #원래값-숫자 목록

# #레이블 데이터 적용
# lab_in=lab_encoder.transform(data_frame['Intent'])   # label 값
# ori_in=lab_encoder.inverse_transform(lab_in)         # 원래 값
# data_frame['Intent'] = lab_in