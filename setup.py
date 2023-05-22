from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='ligo_usecase_3',
    version='0.1',
    packages=['ligo_usecase_3'],
    url='',
    license='MIT',
    author='Chakravarthi Kanduri',
    author_email='chakra.kanduri@gmail.com',
    description='',
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': ['concat_airr=ligo_usecase_3.concat_airr:execute',
                                      'mil_classifier=ligo_usecase_3.multiple_instance_learning_classifier:execute'
                                      ]})