# DataPrediction

This is the back-end part of the DataPrediction program.

Usage:

1. Install package dependencies:

    You can use `anaconda` or `miniconda` to install package dependencies. Paste this yaml file and rename it to `Django.yaml`, then use `conda create -f Django.yaml`.

    ```yaml
    name: Django
    channels:
      - defaults
    dependencies:
      - ca-certificates=2023.08.22=haa95532_0
      - openssl=3.0.12=h2bbff1b_0
      - pip=23.3=py39haa95532_0
      - python=3.9.18=h1aa4202_0
      - setuptools=68.0.0=py39haa95532_0
      - sqlite=3.41.2=h2bbff1b_0
      - vc=14.2=h21ff451_1
      - vs2015_runtime=14.27.29016=h5e58377_2
      - wheel=0.41.2=py39haa95532_0
      - pip:
          - amqp==5.2.0
          - asgiref==3.7.2
          - async-timeout==4.0.3
          - attrs==23.1.0
          - autobahn==23.6.2
          - automat==22.10.0
          - billiard==4.2.0
          - celery==5.3.6
          - certifi==2022.12.7
          - cffi==1.16.0
          - channels==4.0.0
          - charset-normalizer==2.1.1
          - click==8.1.7
          - click-didyoumean==0.3.0
          - click-plugins==1.1.1
          - click-repl==0.3.0
          - colorama==0.4.6
          - constantly==23.10.4
          - contourpy==1.2.0
          - cryptography==41.0.7
          - cycler==0.12.1
          - daphne==4.0.0
          - django==4.2.7
          - djangorestframework==3.14.0
          - filelock==3.9.0
          - fonttools==4.46.0
          - fsspec==2023.4.0
          - hyperlink==21.0.0
          - idna==3.6
          - importlib-resources==6.1.1
          - incremental==22.10.0
          - jinja2==3.1.2
          - joblib==1.3.2
          - kiwisolver==1.4.5
          - kombu==5.3.4
          - markupsafe==2.1.3
          - matplotlib==3.8.2
          - mpmath==1.3.0
          - networkx==3.0
          - numpy==1.26.1
          - packaging==23.2
          - pandas==2.1.2
          - pillow==10.1.0
          - prompt-toolkit==3.0.41
          - pyasn1==0.5.1
          - pyasn1-modules==0.3.0
          - pycparser==2.21
          - pyopenssl==23.3.0
          - pyparsing==3.1.1
          - python-dateutil==2.8.2
          - pytz==2023.3.post1
          - redis==5.0.1
          - requests==2.28.1
          - scikit-learn==1.3.2
          - scipy==1.11.4
          - service-identity==23.1.0
          - six==1.16.0
          - sqlparse==0.4.4
          - sympy==1.12
          - threadpoolctl==3.2.0
          - torch==2.1.1+cu118
          - torchaudio==2.1.1+cu118
          - torchvision==0.16.1+cu118
          - twisted==23.10.0
          - twisted-iocpsupport==1.0.4
          - txaio==23.1.1
          - typing-extensions==4.8.0
          - tzdata==2023.3
          - urllib3==1.26.13
          - vine==5.1.0
          - wcwidth==0.2.12
          - zipp==3.17.0
          - zope-interface==6.1
    prefix: E:\Users\19535\miniconda3\envs\Django
    ```

    And activate the environment with `conda activate Django`.

    You can also use `pip install -r requirements.txt` to install these packages, or install them one by one manually when trying to start the server.

2. Run **celery** under the root directory using `celery -A DataPrediction worker --loglevel=info --pool=solo  --concurrency=1` (for Windows).
3. Run a [redis server](https://github.com/tporadowski/redis/releases). You may need to change the address, port number and password of the redis server in `utils/db.py`.   
4. Run `python manage.py runserver`. The service will run on port 8000 by default.
5. Run the front-end server.