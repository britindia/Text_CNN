## Sentiment Analysis with Tensotflow

Aim: 
1.	Train a Convolution Neural Network (CNN) ML prediction model using Tensorflow predicting sentiment (Positive/Negative) of movie reviews posted on RottenTomato.
2.	Train locally and on GCP(AI Platform)
3.	Perform Hyper-tuning and monitor model performance using tensorboard
4.	Save model protobuff file (crate .pb file)
5.	Serve predictions from .pb file
6.	Package model as Online live application using Flask and host on GCP (App Engine)


### Download and set-up model for training

Download model from - [https://github.com/britindia/Text_CNN/tree/master/trainer]

I highly recommend creating virtual environment to test and train the model. Virtual env help keep all dependencies together. 
I used Python 3.5 for this project.

Steps to create virtual env:
```markdown
-	pip install --user --upgrade virtualenv
-	virtualenv <name>
-	source <name>/bin/activate

```
Install dependencies:
```markdown
-	pip install tensorflow==1.15
-	List of packages in virtualenv: pip list
-	To capture dependencies-  pip freeze -local > requirements.txt
```

Command to run on Mac- 
Make sure you are in Text_CNN folder when you run the model. Directory structure should be same as in GitHub.
```markdown
train-  ./trainer/task.py
eval - ./trainer/eval.py --eval_train --
```
checkpoint_dir=./trainer/runs/1594310095/checkpoints

Make sure to run the model locally first before you attempt to host it on GCP.

tensorboard --logdir=runs/1595596267/summaries

Accuracy = True Predictions/ All Predictions

### Host model on GCP

Create bucket for your project:
```markdown
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
REGION=us-west1
```
For me: BUCKET_NAME =’gs://ordinal-reason-282519-aiplatform’

Install and initialize the Cloud SDK:
```markdown
See the steps in below link -  https://cloud.google.com/sdk/docs
```

Use below commands to initialize gcloud- 
```markdown
type ‘gcloud init’ and provide parameter values
```

Some useful commands to interact with you bucket using gsutil.
```markdown
gsutil mb -l $REGION gs://$BUCKET_NAME
Copy files to GCP - gsutil cp -r Text_CNN gs://$BUCKET_NAME
```

To run model:
```markdown
JOB_NAME="text_cnn_training_"$(date '+%s')
BUCKET_NAME='ordinal-reason-282519-aiplatform'
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-west1

In Single mode:
gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.15 \
    --python-version 3.5 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \


In distributed mode:
JOB_NAME="text_cnn_training_"$(date '+%s')
BUCKET_NAME='ordinal-reason-282519-aiplatform'
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
REGION=us-west1
```

Tensorboard:
```markdown
tensorboard --logdir=$OUTPUT_PATH
```
### Download App Engine code

Download code from https://github.com/britindia/Text_CNN/tree/master/online_prediction

Prerequisites: Must have main.py and app.yaml file.

To deploy app in gCloud:
```markdown
gcloud app deploy app.yaml
```
