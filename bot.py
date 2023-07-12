# pip install botogram

import botogram
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

df = pd.read_csv('./sms_dataset.tsv', sep='\t')
X = df.message
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = Pipeline([('tfidf', TfidfVectorizer()), ('cls', LinearSVC())])
model.fit(X_train, y_train)

bot = botogram.create("YOUR TOKEN")

@bot.message_matches(r'.+')
def pridict_message(chat, message, matches):
    predicted = model.predict([message.text])[0]
    msg = f"This is a {predicted} message " + ('✅'  if predicted == 'ham' else '🚫')
    chat.send(msg)

if __name__ == "__main__":
    bot.run()
