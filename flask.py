#!/usr/bin/env python
# coding: utf-8

# In[163]:


from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np


# In[168]:


app = Flask(__name__,template_folder=r"C:\Users\SANYA\Desktop\work\Hackathon BVIP")

model=pickle.load(open('model.pkl','rb'))


# In[169]:


@app.route('/')
def hello_world():
    return render_template('input.html.html')


# In[170]:


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    f=np.array(int_features)
    prediction=model.predict([f])
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    return render_template('input.html.html')
    

    


# In[ ]:


if __name__ == "__main__":
   from werkzeug.serving import run_simple
   run_simple('localhost', 9000, app)


# In[ ]:





# In[ ]:




