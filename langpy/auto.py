def get_langpy(api_key='', platform='gpt', model='gpt-4'):
    try:
        import google.colab
        from .colab_lp import ColabLangPy
        autolp = ColabLangPy(api_key, platform=platform, model=model)
    except:
        from .jupyter_lp import JupyterLangPy
        autolp = JupyterLangPy(api_key, platform=platform, model=model)
    return autolp
