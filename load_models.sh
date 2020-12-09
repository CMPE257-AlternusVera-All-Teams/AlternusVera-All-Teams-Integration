if [ -d "./all_models" ]
then 
    echo "Directory all models already exists"
else
    mkdir all_models
    wget "https://cmpe-257-alternus-vera.s3.amazonaws.com/feature_finders_toxicity_model"
    mv feature_finders_toxicity_model ./all_models/
fi