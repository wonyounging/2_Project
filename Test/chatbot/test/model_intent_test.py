from chatbot.Preprocess2 import Preprocess2
from chatbot.IntentModel import IntentModel

p = Preprocess2(word2index_dic='c:/2nd_project/Data/chatbot_dict.bin',
                userdic='c:/2nd_project/Data/user_dic.txt')

intent = IntentModel(model_name='c:/2nd_project/Model/intent_model_0807_d.h5', proprocess=p)

items=['오늘 영화 추천해줘']

for item in items:
    predict = intent.predict_class(item)
    predict_label = intent.labels[predict]

    print(item)
    print("의도 예측 클래스 : ", predict)
    print("의도 예측 레이블 : ", predict_label)