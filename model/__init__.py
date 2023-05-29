import pickle


# source: https://stackoverflow.com/questions/62932368/best-way-to-save-many-tensors-of-different-shapes
def save_tensor(tensor, path):
    """
    dump a tensor into a file
    """
    with open(path, 'wb') as f:
        pickle.dump(tensor, f)
        f.close()


def load_tensor(path):
    """
    load a
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        tensor = pickle.load(f)
        f.close()
        return tensor
