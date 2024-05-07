FROM modelstore-base

# Copy library source
COPY modelstore ./modelstore
COPY tests ./tests

# Run tests
ENTRYPOINT ["python3", "-m", "pytest", "--exitfirst", "./tests"]
