import torch
from transformers import BertTokenizer, BertForSequenceClassification
from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd

class SensaEncoder():
  def __init__(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    #TODO: move to a script in requirements
    gdd.download_file_from_google_drive(file_id='1SpfmiCq2a2aXTXvFW6cHnm-0eBCpcyxY',
                                  dest_path='./sensationalism_BERT_best.model',
                                  unzip=False)
    
    # Load pre-trained model (weights)                                  
    label_dict = {'Barely sensationalist': 1, 'Not sensationalist': 0, 'Sensationalist': 2}
    self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      # output_attentions=False,
                                                      output_hidden_states=True)
    # Fine tune BERT for sensationalism
    self.model.load_state_dict(torch.load('sensationalism_BERT_best.model', map_location=self.device))
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    self.model.eval()
    
  def encodeText(self, text, dimension=768):
    # Add BERT  the special tokens.
    text = (text[:500]) if len(text) > 512 else text
    marked_text = "[CLS] " + text + " [SEP]"
    
    # Split the sentence into tokens.
    tokenized_text = self.tokenizer.tokenize(marked_text)
    
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
    
    # Mark each token as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through Sensa-BERT, and collect all of the hidden states
    # produced from all 12 layers.
    with torch.no_grad():
      outputs = self.model(tokens_tensor, segments_tensors)
      hidden_states = outputs[1]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0) # torch.Size([13, 1, 22, 768])

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Get the second last tensor. This is where the tokens are store
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding[:dimension]

  def encodedLiarLiar(self, dataset, dimension=768):
    statements = dataset.Statement.tolist()
    encoded_statements = []
    for statement in statements:
      a = self.encodeText(statement, dimension=dimension)
      encoded_statements.append(a.numpy())
    
    df = pd.DataFrame()

    for a in encoded_statements:
      df = df.append({'a':a[0],'b':a[1],'c':a[2],'d':a[3],'e':a[4],
                      'f':a[5],'g':a[6],'h':a[7],'i':a[8],'j':a[9],
                      'k':a[10],'l':a[11],'m':a[12],'n':a[13],'o':a[14],
                      'p':a[15]}, ignore_index=True)
    
    return df

  def getEncodedLiarLiar(self):
    df = pd.read_csv('./nlp_sensationalism/liarliarEncodedTrain.csv')
    return df
