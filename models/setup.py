# Copyright 2022 The PEACH Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install Peach."""

import setuptools

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='peach',
    version='0.0.1',
    description='PEACH',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    install_requires=[
        'absl-py',
        'mock',
        'numpy',
        'rouge-score',
        'sacrebleu',
        'sentencepiece',
        'tensorflow-text==1.15.0rc0',
        'tfds-nightly',
        'tensor2tensor==1.15.0',
        'tensorflow-gpu==1.15.2',
    ],
    extras_require={
        'tensorflow': ['tensorflow-gpu==1.15.2'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='deeplearning machinelearning nlp translation transformer pretraining',
)
