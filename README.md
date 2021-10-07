
### Deploying a Scalable ML Pipeline in Production

Deploy a machine learning model on Heroku. Use Git and DVC to track code, data and model while developing a simple classification model on the Census Income Data Set. After creating the model, the finalize the model for production by checking its performance on slices and writing a model card encapsulating key knowledge about the model. Implement a Continuous Integration and Continuous Deployment framework and ensure pipeline passes a series of unit tests before deployment. Lastly, an API will be written using FastAPI and tested locally. After successful deployment, the API will be tested live using the requests module.

## Execution

Model training
Model training and test can be done by python main.py --choice train_model

Model score
Check score on latest dvs saved model can be done by python main.py --choice get_score

Run entire pipeline
To run the entire pipeline in sequence, use python main.py --choice all

Test API
If testing FastAPi serving on local is needed, execute uvicorn app_server:app --reload

Check Heroku deployed API
Check Heroku deployed API using python heroku_api_test.py

## CI/CD
Every new commit triggers a test pipeline, which triggers pull from DVC and exectute Pytest and Flake8 with Github actions.  

