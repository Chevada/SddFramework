# SddFramework
This repository introduces the SDD framework, which is applied to the pre-trained code model of the encoder-decoder architecture for class-level code generation tasks.

# Before training
## 1.Dependency
  Our project is based on CodeT5, so you can configure the environment according to the description of the CodeT5(https://github.com/salesforce/CodeT5/tree/main/CodeT5) project.
## 2. Data
  The data needed to reproduce our project can be obtained in the PyClass/pyclass path of this repository.
## 3. Base Model
   Our project is based on CodeT5 and UniXcoder, but our project does not provide the corresponding model files. You need to go to the model repository to load the base model yourself.

# Training
  Although our framework conducts class-level code generation experiments based on two different models, the overall training idea is the same.
## 1. Fine-tune the base model
  First, we need to fine-tune our baseline model on the class-level code generation task to enable it to have basic class-level code generation capabilities.
  Specifically, in the codet5 project, execute the 'run_pyclass_gen.py' file. For specific execution methods, refer to the CodeT5/command.txt file in this repository;
  In the UniXcoder project, execute the 'run_gen_class.py' file. The specific instructions are described in the UniXcoder/code-generation/README.md file in this repository.
## 2. Applying the SDD framework
  ### 1. Using the fine-tuned baseline model above, generate and cache data for training the syntax discriminator. 
  Specifically, execute the 'run_gen_use_sia_model_gen_ast_loss_data.py' file in the CodeT5 project; execute the 'run_gen_ast_loss_data.py' file in the UniXcode project.
  ### 2. Training the syntax discriminator. 
  Specifically, execute the 'run_gen_use_sia_model_train_ast_loss.py' file in the CodeT5 project; execute the 'train_ast_loss.py' file in the UniXcoder project.
  ### 3. Combined with the syntax discriminator for further code generation training. 
  Specifically, execute the 'run_gen_use_sia_model_with_ast.py' file in the CodeT5 project; execute the 'run_sia_gen_with_ast_loss.py' file in the UniXcoder project.
   
