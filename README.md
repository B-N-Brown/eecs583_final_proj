# eecs583_final_proj


# SETUP:

## Environment (version 3.11.2)
### For some reason I had to install this (Debian)
sudo apt-get install libtinfo5

### (MAC)
brew install libtinfo5

python3 -m venv env
pip3 install -r requirements.txt


Also gotta change line 227 of env/lib/python3.11/site-packages/shimmy/openai_gym_compatibility.py to 

self.gym_env.action_space.seed(seed)


Also gotta change line 42 in env/lib/python../site-packages/compiler_gym/wrappers/core.py to 

return self.env.step(action)


## Run
python3 test_model.py


