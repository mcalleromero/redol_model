FILE=VERSION
VERSION=$(shell cat $(FILE))

INPUT_DATA=input_data/

make_dirs:
	mkdir $(INPUT_DATA) || true

all: make_dirs
	docker image build -t redol_model_reader:$(VERSION) .

	docker run -e INPUT_DATA=$(INPUT_DATA) \
			   -v $(PWD)/$(INPUT_DATA):/opt/$(INPUT_DATA) \
			   --name redol_model_reader \
			   -d redol_model_reader:$(VERSION);

clean:
	docker stop redol_model_reader || true
	docker rm redol_model_reader || true
	docker rmi redol_model_reader:$(VERSION) || true