cpu: 
	rm -rf ./submission/y
	rm -rf submission.txt
	rm -rf subby.zip
	mkdir ./submission/y
	cp ./submission/x/* ./submission/y
	python trainer_ash.py -d cpu
	python ../reference/submission.py -s ./submission/y -d ./ -nu 255
	zip subby.zip submission.txt 

cuda: 
	rm -rf ./submission/y
	rm -rf submission.txt
	rm -rf subby.zip
	mkdir ./submission/y
	cp ./submission/x/* ./submission/y
	python trainer_ash.py -d cuda
	python ../reference/submission.py -s ./submission/y -d ./ -nu 255
	zip subby.zip submission.txt 

clean: 
	rm -rf ./submission/y
	rm -rf submission.txt
	rm -rf subby.zip
	mkdir ./submission/y
	cp ./submission/x/* ./submission/y


sub: 
	cp ./submission/x/* ./submission/y

	python ../reference/submission.py -s ./submission/y -d ./ -nu 255
	zip subby.zip submission.txt 
