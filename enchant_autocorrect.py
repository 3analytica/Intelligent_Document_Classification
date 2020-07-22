import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter

d = enchant.Dict("el_GR")

def enchant_correction(word):
    try:
        return d.suggest(word)[0]    
    except (IndexError, TypeError):
        return word

    
    
    