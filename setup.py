from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='intgrads',
    entry_points={
        'console_scripts': [
            'intgrads-images=intgrads_images.serve:serve'
        ],
    },
    install_requires=[
        "captum==0.2.0",
        "click",
        "click_completion",
        "logbook",
        "flask",
        "torch",
        "torchvision",
    ],
)
