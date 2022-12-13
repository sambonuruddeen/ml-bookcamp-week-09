# ML-Zoomcamp-Week-09
[Serverless Deeplearning](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless)

Deployed a Machine Learning Service to classify images as Dinosuars or Dragon.

The serverless service leverages the AWS Lambda to serve the predictions to users.

The code is finally packaged in as Docker Image for easy deployment.

# How to Run

1- Download this Repository

2- Open Terminal in the project directory

3- Buil the Image with the command

`docker build -t dino-model .`

4- When the image is finished building, run the  command below

`docker run -it --rm -p 8080:8080 dino-model:latest`

5- Open another terminal window in the project directory and invoke `python3 test.py`

**Change the value of the `data` dictionary inside `test.py` with that of your own image to perform prediction on it**

`data = {'url': 'url-of-your-image'}`

