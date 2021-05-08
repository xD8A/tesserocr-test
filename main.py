from PIL import Image


def is_cyrillic(symbol):
    return 'а' <= symbol.lower() <= 'я' or symbol in (' ', '.')


def main(image: Image):
    from warnings import warn
    from math import prod
    from itertools import chain, product
    from tesserocr import PyTessBaseAPI

    with PyTessBaseAPI(lang='rus') as api:
        api.SetImage(image)
        api.SetVariable('lstm_choice_mode', '2')
        if not api.Recognize(timeout=60):
            warn('cannot recognize image')

        choices = api.GetBestLSTMSymbolChoices()
        choices = list(chain.from_iterable(choices))
        cases = dict()
        for i, symbol_choice in enumerate(choices):
            symbol_choice = [(symbol, prob) for symbol, prob in symbol_choice if is_cyrillic(symbol) and prob > 0.05]
            if not symbol_choice:
                symbol_choice = [('_', 1)]
            symbols, probs = zip(*symbol_choice)
            symbols = ''.join(symbols)
            if cases:
                cases = {''.join(k): prod(v) for k, v in zip(product(cases.keys(), symbols),
                                                             product(cases.values(), probs))}
            else:
                cases = dict(symbol_choice)
        cases = sorted(cases.items(), key=lambda itm: -1*itm[1])
        for word, prob in cases:
            print(word, f'(prob = {prob})')


if __name__ == '__main__':
    main(Image.open('nn.png'))
