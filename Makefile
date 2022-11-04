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

cleanset2:
	rm -rf ./dataset2/images
	rm -rf ./dataset2/ground_truths
	rm -rf ./dataset2/test
	mkdir ./dataset2/images
	mkdir ./dataset2/ground_truths
	mkdir ./dataset2/test

sub: 
	rm -rf ./submission/y
	rm -rf submission.txt
	rm -rf subby.zip
	mkdir ./submission/y
	cp ./submission/x/* ./submission/y
	python trainer_ash.py -d cuda
	python ../reference/submission.py -s ./submission/y -d ./ -nu 255
	zip subby.zip submission.txt 

cleanset2.1:
	rm -rf ./dataset2.1/images
	rm -rf ./dataset2.1/ground_truths
	rm -rf ./dataset2.1/test
	mkdir ./dataset2.1/images
	mkdir ./dataset2.1/ground_truths
	mkdir ./dataset2.1/test


cleanset3:
	rm -rf ./dataset3/images
	rm -rf ./dataset3/ground_truths
	rm -rf ./dataset3/test
	mkdir ./dataset3/images
	mkdir ./dataset3/ground_truths
	mkdir ./dataset3/test

cleanset4:
	rm -rf ./dataset4/images
	rm -rf ./dataset4/ground_truths
	rm -rf ./dataset4/test
	mkdir ./dataset4/images
	mkdir ./dataset4/ground_truths
	mkdir ./dataset4/test

cleanset11:
	rm -rf ./dataset11/images
	rm -rf ./dataset11/ground_truths
	rm -rf ./dataset11/test
	mkdir ./dataset11/images
	mkdir ./dataset11/ground_truths
	mkdir ./dataset11/test

cleanset12:
	rm -rf ./dataset12/images
	rm -rf ./dataset12/ground_truths
	rm -rf ./dataset12/test
	mkdir ./dataset12/images
	mkdir ./dataset12/ground_truths
	mkdir ./dataset12/test