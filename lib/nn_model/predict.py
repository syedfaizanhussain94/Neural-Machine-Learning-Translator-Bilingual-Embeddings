import numpy as np

from utilities.utilities import tokenize
from lib.w2v_model.vectorizer import get_token_vector
from lib.dialog_processor import EOS_SYMBOL, EMPTY_TOKEN
from configs.config import TOKEN_REPRESENTATION_SIZE, TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH
from utilities.utilities import get_logger
from scipy import spatial
_logger = get_logger(__name__)


def _sequence_to_vector(sentence, w2v_model):
    # Here we need predicted vectors only for one sequence, not for the whole batch,
    # however StatefulRNN works in a such a way that we have to feed predict() function
    # the same number of examples as in our train batch.
    # Then we can use only the first predicted sequence and disregard all the rest.
    # If you have more questions, feel free to address them to https://github.com/farizrahman4u/seq2seq
    X = np.zeros((TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE))

    for t, token in enumerate(sentence):
        X[0, t] = get_token_vector(token, w2v_model)

    return X


def _is_good_token_sequence(token_sequence):
    return EMPTY_TOKEN not in token_sequence and token_sequence[-1] == EOS_SYMBOL

def compute_similarities(prediction_vector,w2v_model,index_to_token):
    similarities=[]
    for i in range(len(index_to_token)):
        similarity=1 - spatial.distance.cosine(get_token_vector(index_to_token[i],w2v_model),prediction_vector)
        similarities.append(similarity)
    return similarities

def _predict_sequence(input_sequence, nn_model, w2v_model, index_to_token, diversity):
    input_sequence = input_sequence[:ANSWER_MAX_TOKEN_LENGTH]

    X = _sequence_to_vector(input_sequence, w2v_model)
    predictions = nn_model.predict(X, verbose=0)[0]
    # print "shape",np.shape(predictions)
    predicted_sequence = []

    for prediction_vector in predictions:
        cosine_distances = compute_similarities(prediction_vector,w2v_model,index_to_token  )
        next_index = np.argmax(cosine_distances)
        # next_index = np.argmax(prediction_vector)
        next_token = index_to_token[next_index]
        predicted_sequence.append(next_token)

    return predicted_sequence


def predict_sentence(sentence, nn_model, w2v_model, index_to_token, diversity=0.5):
    input_sequence = tokenize(sentence + ' ' + EOS_SYMBOL)
    tokens_sequence = _predict_sequence(input_sequence, nn_model, w2v_model, index_to_token, diversity)
    predicted_sentence = ' '.join(tokens_sequence)

    return predicted_sentence