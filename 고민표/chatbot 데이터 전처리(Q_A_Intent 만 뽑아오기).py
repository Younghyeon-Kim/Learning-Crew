import pandas as pd
import os

path = "C:/Users/user/Desktop/data/chatbot_data/"
#파일 불러오기(경로명은 각자)
df = pd.read_csv(path + "카페_validation.csv")
os.getcwd()

#c-a , s-q 지우기
no_data1 = df[ (df['발화자'] == 's') & (df['QA여부'] == 'q')].index
no_data2 = df[ (df['발화자'] == 'c') & (df['QA여부'] == 'a')].index
df1 = df.drop(no_data1)
df1.drop(no_data2, inplace = True)
df1.reset_index(inplace = True, drop = True)


#인텐트 항목 앞글자만 자르기
lst = list(df1['인텐트'])
lst2 = [i.split('_')[0] for i in lst]
df1['인텐트'] = lst2


# qa 정렬대로 뽑아오기
Q = []
A = []
Intent = []

for i in df1.index:
  if df1['QA여부'][i] == 'q' and df1['QA여부'][i+1] == 'a':
    Q.append(df1.loc[i]['발화문'])
    A.append(df1.loc[i+1]['발화문'])
    Intent.append(df1.loc[i]['인텐트'])

cafe_data= pd.DataFrame({'Q':Q,
                'A':A,
                'Intent':Intent})


cafe_data

cafe_data.to_csv(path + 'cafe_qa_validation.csv')

pd.read_csv('cafe_qa_validation.csv')