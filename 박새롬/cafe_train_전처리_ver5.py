# -*- coding: utf-8 -*-
"""카페_전처리.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aH4SB1OgaTfVzMSayWNnMLxd2sQSGDLZ
"""

import pandas as pd

cafe_data = pd.read_csv('/content/drive/MyDrive/미니 프로젝트/data/카페_train.csv')

df = cafe_data.dropna(axis=1)
df.head()

# 직원이 질문하는 경우 & 소비자가 대답하는 경우 삭제

x1 = df[(df['발화자']=='s')&(df['QA여부']=='q')].index
x2 = df[(df['발화자']=='c')&(df['QA여부']=='a')].index

df = df.drop(x1)
df = df.drop(x2)
df.reset_index(inplace = True, drop = True)

# QA 정렬대로 뽑아오기
Q = []
A = []
label = []

for i in df.index:
  if df['QA여부'][i] == 'q' and df['QA여부'][i+1] == 'a':
    Q.append(df.loc[i]['발화문'])
    A.append(df.loc[i+1]['발화문'])
    label.append(df.loc[i]['인텐트'])

data= pd.DataFrame({'Q':Q,
                'A':A,
                'label' :label})


data

# ===========================================
# #############   데이터 삭제   ###############
# ===========================================
# 중복 질문 제거
# -------------------------------------------

# 데이터: 72559 --> 중복 제거 후 데이터: 66041 
data.drop_duplicates(['Q'], inplace = True)
data.reset_index(inplace = True, drop = True)

# -------------------------------------------
# label 삭제: 카페에 필요한 데이터만 남기기
# 매장, 멤버십, 예약, 웹사이트, AS, 제품_추천, 제품_용도, 비교, 재고, 교환, 제품_가격 삭제
# -------------------------------------------

del_label = data[data['label'].str.contains('매장|멤버십|예약|웹사이트|AS|제품_추천|제품_용도|비교|재고|교환|제품_가격')].index
data = data.drop(del_label)

data.reset_index(inplace = True, drop = True)

# -------------------------------------------
# 필요 없는 데이터 삭제
# -------------------------------------------

# 배달/배송 관련 정보 삭제
delivery_q = data[data['Q'].str.contains('배달|택배')].index
data = data.drop(delivery_q)

delivery_a = data[data['A'].str.contains('배달|택배')].index
data = data.drop(delivery_a)

delivery = data[data['label'].str.contains('배송')].index
data = data.drop(delivery)


# 수거 삭제
waste_q = data[data['Q'].str.contains('수거')].index
data = data.drop(waste_q)

waste_a = data[data['A'].str.contains('수거')].index
data = data.drop(waste_a)



# 특정 상표명, 카페 데이터 삭제
extra_q = data[data['Q'].str.contains('설빙|베스킨라빈스|스타벅스|공차|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|하프갤론|텀블러')].index
data = data.drop(extra_q)

extra_a = data[data['A'].str.contains('설빙|베스킨라빈스|스타벅스|공차|나무|단청|도마|그림|파인트|쿼터|패밀리|하프갤런|하프갤론|텀블러')].index
data = data.drop(extra_a)



# 기타 특수 데이터 삭제
dummy = data[data['A'].str.contains('오류')].index
data = data.drop(dummy)

bill = data[data['Q'].str.contains('영수증 버|영수증은 버|영수증버|영수증은버')].index
data = data.drop(bill)

price = data[data['Q'].str.contains('얼마') & data['label'].str.contains('결제_일반')].index
data = data.drop(price)

nutrient = data[data['Q'].str.contains('영양성분')].index
data = data.drop(nutrient)

nickname = data[data['A'].str.contains('닉네임')].index
data = data.drop(nickname)

etc_q = data[data['Q'].str.contains('불면증|데이터|오늘의 커피|십만원권 이서')].index
data = data.drop(etc_q)
etc_a = data[data['A'].str.contains('서비스')].index
data = data.drop(etc_a)



# 결제 관련 특수 데이터 삭제
kakaopay1 = data[data['label'].str.contains('결제_수단') & data['Q'].str.contains('카카오 페이|카카오페이') & data['A'].str.contains('안')].index
data = data.drop(kakaopay1)
kakaopay2 = data[data['label'].str.contains('결제_수단') & data['A'].str.contains('재고')].index
data = data.drop(kakaopay2)

card1 = data[data['label'].str.contains('결제_수단') & data['Q'].str.contains('현금') & data['A'].str.contains('카드전용|카드 전용')].index
data = data.drop(card1)
card2 = data[data['label'].str.contains('결제_수단') & data['Q'].str.contains('현금') & data['A'].str.contains('카드 기계')].index
data = data.drop(card2)

negative_answ = data[data['label'].str.contains('결제_수단') & data['A'].str.contains('아니오|아니요')].index
data = data.drop(negative_answ)

account = data[data['A'].str.contains('계좌번호')].index
data = data.drop(account)

local_card = data[data['label'].str.contains('결제_수단') & data['Q'].str.contains('카드') & data['A'].str.contains('불가능')].index
data = data.drop(local_card)

negative_answ_card = data[data['label'].str.contains('결제_수단') & data['Q'].str.contains('카드') & data['A'].str.contains('생일쿠폰|죄송')].index
data = data.drop(negative_answ_card)


data.reset_index(inplace=True, drop=True)

# -------------------------------------------
# 답변할 수 없는 데이터 삭제
# -------------------------------------------

# 날짜 묻는 질문/날짜로 대답하는 답변 삭제
product_date_q = data[data['Q'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_q)

product_date_a = data[data['A'].str.contains('1월|2월|3월|4월|5월|6월|7월|8월|9월|10월')].index
data = data.drop(product_date_a)


# 시간 관련 질문/답변 삭제
time_q = data[data['label'].str.contains('제품_정보') & data['Q'].str.contains('몇 분|시간|몇분')].index
data = data.drop(time_q)

time_a = data[data['label'].str.contains('제품_정보') & data['A'].str.contains('몇 분|시간|몇분')].index
data = data.drop(time_a)

# 총 ~에요? 들어가는 질문 삭제
total_q = data[data['Q'].str.contains('총 얼마예요?|총 얼마')].index
data = data.drop(total_q)

total_sum = data[data['Q'].str.contains('총 몇')].index
data = data.drop(total_sum)


# 금액 안내 답변 삭제(카페에 맞게끔 새로운 데이터 추가 필요)
price_answer = data[data['A'].str.contains('원입니다|원 입니다')].index
data = data.drop(price_answer)


# 원두 삭제
coffee_q = data[data['Q'].str.contains('원두')].index
data = data.drop(coffee_q)

coffee_a = data[data['A'].str.contains('원두')].index
data = data.drop(coffee_a)


data.reset_index(inplace=True, drop=True)

# -------------------------------------------
# 현재 카페에 맞게끔 데이터 삭제
# -------------------------------------------

# 품절 삭제
absence_q = data[data['Q'].str.contains('품절')].index
data = data.drop(absence_q)

absence_a = data[data['A'].str.contains('품절')].index
data = data.drop(absence_a)



# 세트 메뉴 삭제
setmenu_q = data[data['Q'].str.contains('세트')].index
data = data.drop(setmenu_q)

setmenu_a = data[data['A'].str.contains('세트')].index
data = data.drop(setmenu_a)

data.reset_index(inplace=True, drop=True)



# 신메뉴 삭제
new_menu = data[data['label'].str.contains('제품_정보') & data['Q'].str.contains('신메뉴|새로')].index
data = data.drop(new_menu)



# 밥, 떡, 요거트 삭제
food_q = data[data['Q'].str.contains('밥|떡|요거트')].index
data = data.drop(food_q)

food_a = data[data['Q'].str.contains('밥|떡|요거트')].index
data = data.drop(food_a)

# ===========================================
# #############   데이터 학습   ###############
# ===========================================
# -------------------------------------------
# 결제 수단 학습
# 
# 가능: 현금, 카드, 모바일 페이
# 카운터 문의: 상품권, 분할 결제
# 불가: 포인트, 기프티콘, 할인
# -------------------------------------------

data.loc[data['label'].str.contains('결제_수단') & data['Q'].str.contains('페이') & data['Q'].str.contains('가능'), 'A'] = '네, 모든 모바일 페이를 지원하고 있습니다.'
data.loc[data['label'].str.contains('결제_수단') & data['Q'].str.contains('제로페이') & data['Q'].str.contains('되나요|있나요|돼요|있을까요|가능하죠|거죠|맞아요'), 'A'] = '네, 모든 모바일 페이를 지원하고 있습니다.'
data.loc[data['Q'].str.contains('어떤') & data['Q'].str.contains('페이'), 'A'] = '모든 모바일 페이를 지원하고 있습니다.'

data.loc[data['label'].str.contains('결제_수단') & data['Q'].str.contains('현금만'), 'A'] = '현금, 카드 모바일 페이로 결제 가능합니다.'

data.loc[data['Q'].str.contains('상품권'), 'A'] = '상품권 사용은 카운터에 문의 바랍니다.'

data.loc[data['Q'].str.contains('포인트'), 'A'] = '현재 포인트 기능은 제공되지 않습니다.'

data.loc[data['Q'].str.contains('기프티콘|기포티콘|쿠폰'), 'A'] = '저희 카페는 쿠폰과 기프티콘이 없습니다.'

# 분할 결제 불가 안내 문구 학습
data.loc[data['Q'].str.contains('분할 결제|분할결제|나눠서 결제|따로 결제'), 'A'] = '분할 결제는 카운터에 문의 바랍니다.'
data.loc[data['Q'].str.contains('두 개|두 가지|동시에|나머지|둘 다') & data['label'].str.contains('결제_수단'), 'A'] = '분할 결제는 카운터에 문의 바랍니다.'
data.loc[data['label'].str.contains('결제_수단') & data['Q'].str.contains('카드로') & data['Q'].str.contains('현금으로'), 'A'] = '분할 결제는 카운터에 문의 바랍니다.'

# 할인 불가 안내 문구 학습
data.loc[data['Q'].str.contains('할인|깎|깍|제휴|DC'), 'A'] = '현재 할인은 불가합니다.'

# -------------------------------------------
# 기타 학습
# 
# 카운터 문의: 오류/불량, 칼로리 정보
# 불가: 재료 변경, 디카페인
# 등
# -------------------------------------------

# 재료 변경 안내 학습
data.loc[data['Q'].str.contains('빼|교환|바꾸|추가|바꿔|뺄|변경|넣지 말|넣지 마|많이|말고|반만|반 만|조금만 더|더 넣어|올려') & data['label'].str.contains('제품_구성|제품_소재|주문_제품'), 'A'] = '구성 관련 요청은 카운터에 문의 바랍니다.'
data.loc[data['label'].str.contains('제품_정보') & data['Q'].str.contains('토핑추가|토핑 추가'), 'A'] = '구성 관련 요청은 카운터에 문의 바랍니다.'
data.loc[data['Q'].str.contains('두유') & data['label'].str.contains('제품_구성|제품_소재'), 'A'] = '현재 두유는 제공되지 않습니다.'
data.loc[data['Q'].str.contains('저지방 우유|저지방우유|무지방우유|무지방 우유'), 'A'] = '현재 저지방/무지방 우유는 제공되지 않습니다.'
data.loc[data['Q'].str.contains('당도'), 'A'] = '당도 단계 선택은 불가능합니다.'
data.loc[data['Q'].str.contains('샷 추가|샷추가') & data['label'].str.contains('제품_구성') & data['Q'].str.contains('나요|죠|가요|까요|해요|돼요'), 'A'] = '샷 추가 가능합니다.'
data.loc[data['Q'].str.contains('샷 추가|샷추가') & data['label'].str.contains('제품_구성') & ~data['Q'].str.contains('나요|죠|가요|까요|해요|돼요'), 'A'] = '샷 추가 해드리겠습니다.'

# 사이즈 관련 답변 학습(현재는 사이즈 1개로 통일)
data.loc[data['Q'].str.contains('사이즈') & ~data['Q'].str.contains('스몰|톨'), 'A'] = '저희 카페는 기본 사이즈인 톨 사이즈로만 제공하고 있습니다.'
data.loc[data['label'].str.contains('제품_정보_질문') & data['Q'].str.contains('그란데|벤티'), 'A'] = '저희 카페는 기본 사이즈인 톨 사이즈로만 제공하고 있습니다.'

# 제품 함량 문의 안내 학습
data.loc[data['Q'].str.contains('함량'), 'A'] = '상세 정보는 카운터에 문의 바랍니다.'
data.loc[data['label'].str.contains('제품_정보') & data['Q'].str.contains('그램'), 'A'] = '상세 정보는 카운터에 문의 바랍니다.'

# 디카페인 불가 안내, 카페인 문의 안내 문구 학습
data.loc[data['Q'].str.contains('디카페인|디 카페인'), 'A'] = '현재 디카페인 음료는 제공되지 않습니다.'
data.loc[data['Q'].str.contains('카페인 함량'), 'A'] = '카페인 함량은 카운터에 문의 바랍니다.'

# 리필 안내 문구 학습
data.loc[data['Q'].str.contains('리필') & data['Q'].str.contains('물|얼음'), 'A'] = '물과 얼음 리필은 카운터에 문의해주세요₍՞◌′ᵕ‵ू◌₎♡'
data.loc[data['Q'].str.contains('리필') & ~data['Q'].str.contains('물|얼음'), 'A'] = '리필은 불가능합니다. 새로 주문해주세요 ʕ”̮ॽु⋆⁺₊⋆ ♡̷̷̷ ' 



# 오류/불량 문의 안내 문구 학습
data.loc[data['label'].str.contains('제품_불량|결제_오류|주문_오류'), 'A'] = '죄송합니다. 카운터에서 확인 도와드리겠습니다.'

# 칼로리 정보 불가능 안내 문구 학습
data.loc[data['Q'].str.contains('열량|칼로리'), 'A'] = '칼로리 정보는 카운터에 문의 바랍니다.'



# 행사 안내 문구 학습
data.loc[data['Q'].str.contains('행사|이벤트'), 'A'] = '현재 진행중인 행사가 없습니다. 다음 행사를 기다려주세요₍՞◌′ᵕ‵ू◌₎♡'
data.loc[data['label'].str.contains('행사'), 'A'] = '현재 진행중인 행사가 없습니다. 다음 행사를 기다려주세요₍՞◌′ᵕ‵ू◌₎♡'

# 포장 학습
data.loc[data['label'].str.contains('포장_비용'), 'A'] = '포장 비용은 추가되지 않습니다.'

# Transformer 모델 사용시 label 제거
# data = data.drop(['label'], axis=1)

# ### KoGPT 모델 사용시 label 필요

# lst = list(data['label'])
# lst2 = [i.split('_')[0] if i.split('_')[0] != '제품' else i.split('_')[0] +'_'+ i.split('_')[2] for i in lst]

# data['label'] = lst2

# data_frame = data

# #################################
# ### label 항목 레이블
# #################################

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

# #레이블 인코딩
# lab_encoder = LabelEncoder()
# lab_encoder.fit(data_frame['label'].unique())
# lst = lab_encoder.classes_
# label_list = dict(zip(lst, range(len(lst))))

# print(label_list)    #원래값-숫자 목록

# #레이블 데이터 적용
# lab_in=lab_encoder.transform(data_frame['label'])   # label 값
# ori_in=lab_encoder.inverse_transform(lab_in)         # 원래 값
# data_frame['label'] = lab_in

# DATA_PATH = '/content/drive/MyDrive/미니 프로젝트/data/'
# data.to_csv(DATA_PATH + 'cafe_qa_train.csv', index=False)