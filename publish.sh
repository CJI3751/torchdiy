# 例如：$1=0.4.2
pip freeze > requirements.txt
git add -A
git commit -m "v$1"
git push
python setup.py sdist bdist_wheel
twine upload dist/torchdiy-$1.* --verbose
