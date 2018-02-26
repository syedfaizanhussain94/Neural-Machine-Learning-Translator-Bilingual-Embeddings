import sys
import os
from itertools import tee

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dialog_processor import get_processed_dialog_lines_and_index_to_token
from configs.config import CORPUS_PATH_EN, CORPUS_PATH_DE, PROCESSED_CORPUS_PATH_EN,PROCESSED_CORPUS_PATH_DE, TOKEN_INDEX_PATH_EN,TOKEN_INDEX_PATH_DE, W2V_PARAMS_EN,W2V_PARAMS_DE
from lib.w2v_model import w2v
from lib.nn_model.model import get_nn_model
from lib.nn_model.train import train_model
from lib.nn_model.train import train_model_new
from utilities.utilities import get_logger

_logger = get_logger(__name__)


def learn():
    # preprocess the dialog and get index for its vocabulary
    # processed_dialog_lines, index_to_token = \
    #     get_processed_dialog_lines_and_index_to_token(CORPUS_PATH, PROCESSED_CORPUS_PATH, TOKEN_INDEX_PATH)

    processed_dialog_lines_en, processed_dialog_lines_de, index_to_token_en, index_to_token_de = \
        get_processed_dialog_lines_and_index_to_token(CORPUS_PATH_EN, CORPUS_PATH_DE, PROCESSED_CORPUS_PATH_EN, PROCESSED_CORPUS_PATH_DE, TOKEN_INDEX_PATH_EN, TOKEN_INDEX_PATH_DE)

    # dualize iterator
    # dialog_lines_for_w2v, dialog_lines_for_nn = tee(processed_dialog_lines)

    dialog_lines_for_w2v_en, dialog_lines_for_nn_en = tee(processed_dialog_lines_en)
    dialog_lines_for_w2v_de, dialog_lines_for_nn_de = tee(processed_dialog_lines_de)
    _logger.info('-----')

    # use gensim realisatino of word2vec instead of keras embeddings due to extra flexibility
    w2v_model_en = w2v.get_dialogs_model(W2V_PARAMS_EN, dialog_lines_for_w2v_en)
    w2v_model_de = w2v.get_dialogs_model(W2V_PARAMS_DE, dialog_lines_for_w2v_de)

    _logger.info('-----')

    nn_model = get_nn_model(token_dict_size=len(index_to_token_de))
    _logger.info('-----')

    train_model(nn_model, w2v_model_en, w2v_model_de, dialog_lines_for_nn_en,dialog_lines_for_nn_de, index_to_token_en, index_to_token_de)
    # train_model_new(w2v_model_en, w2v_model_de, dialog_lines_for_nn_en,dialog_lines_for_nn_de, index_to_token_en, index_to_token_de)


if __name__ == '__main__':
    learn()
