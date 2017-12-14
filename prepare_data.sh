unzip question-answer-jokes.zip
cat jokes.csv | sed 's/Q: //g' | sed 's/A: //g' | sed 's/\[OC\] //g' > jokes_parsed.csv
mv jokes_parsed.csv jokes.csv

wget "http://nlp.stanford.edu/data/glove.6B.zip"
unzip glove.6B.zip

wget "http://nlp.stanford.edu/data/glove.42B.300d.zip"
unzip glove.42B.300d.zip

wget "http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.commoncrawl.300d.W.pos.vectors.gz"
gzip -d lexvec.commoncrawl.300d.W.pos.vectors.gz

echo 'Manually download GoogleNews-vectors-negative300.bin.gz in this directory from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing'
echo 'Then extract it and execute obtain_word2vec.py'
