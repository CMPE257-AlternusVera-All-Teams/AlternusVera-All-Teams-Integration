if [ -d "./venv" ]
then
    source ./venv/bin/activate
else
    python -m venv ./venv
    source ./venv/bin/activate
fi
python -m spacy download en_core_web_md
export FLASK_APP=wsgi
export FLASK_ENV=development
pip install -r requirements.txt

echo "start the application by running - flask run"