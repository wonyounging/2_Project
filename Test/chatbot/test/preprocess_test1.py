
from chatbot.Preprocess1 import Preprocess1

sent = "내일 오전 10시에 겨울왕국 예약하고 싶어"

p = Preprocess1(userdic='c:/2nd_project/Data/user_dic.txt')

pos = p.pos(sent)

keywords = p.get_keywords(pos, without_tag=False)

print(keywords)