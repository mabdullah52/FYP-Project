FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements_space.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . /app
# the app writes its database and uploaded scans to /app/app_data, so the user must own /app itself
RUN mkdir -p /app/app_data && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["python", "hf_app.py"]
