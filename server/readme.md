# NLEP Server

The complete source code of building an NLEP API is included in this folder by running the following command
```python
uvicorn nlep_server:app --reload --host 0.0.0.0 --port 8000
```
The LangCode agent can be connected to the preferred API endpoint via
```python
# For example, clp is a LangCode agent
clp.config_api_endpoint()
```
and input your url in this format: `http://{HOSTNAME}:{PORT}/items/0`

This server does not set any timeout.