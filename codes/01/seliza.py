import random
import re

rules=[["(.*)hello(.*)",["Hi there. Please state your problem"]],
           ["(.*)name(.*)",["Great, good to know","I am not interested in names"]],
  ["(.*)sorry(.*)",["please don't apologize","Apologies are not necessary","What feelings you have when you apologize?"]],
 ["(.*)",["Very interesting","I am not sure I understand you fully","Please continue",
         "Do you feel strongly about discussing such things?","\\2"]]]

grammar = {
"am": "are",
"was": "were",
"i": "you",
"i'd": "you would",
"i've": "you have",
"i'll": "you will",
"my": "your",
"are": "am",
"you've": "I have",
"you'll": "I will",
"your": "my",
"yours": "mine",
"you": "me",
"me": "you"
}

def correction(word):
  character=word.lower().split()
  for i, j in enumerate(character):
      if j in grammar:
          character[i]=grammar[j]
  return " ".join(character)

def test(sentence):
  for pattern, message in rules:
      match=re.match(pattern,sentence.rstrip(".!"))
      if match:
          response = random.choice(message)
          temp = " " + correction(match.group())
          response2 = re.sub(r"\\2",temp,response)
          return response2
  recall=random.choice(random.choice([r[1] for r in rules]))
  return recall



while True:
  sentence =input("You: ")
  print("JBot: " + test(sentence))
  if sentence == "quit":
        break

