MARKDOWN_FILES := $(wildcard markdowns/*.md)

NOTEBOOK_FILES := $(patsubst markdowns/%.md,notebooks/%.ipynb,$(MARKDOWN_FILES))

all: $(NOTEBOOK_FILES)

notebooks/%.ipynb: markdowns/%.md
	pandoc --resource-path=notebooks/ $< -o $@

clean:
	rm -f $(NOTEBOOK_FILES)