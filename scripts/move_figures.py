import shutil
import os

source_dir = os.path.abspath(os.path.join(os.getcwd(), '../report/figures'))
dest_dir = os.path.abspath(os.path.join(os.getcwd(), 'report/figures'))

print(f"Moving from {source_dir} to {dest_dir}")

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for filename in os.listdir(source_dir):
    src = os.path.join(source_dir, filename)
    dst = os.path.join(dest_dir, filename)
    if os.path.isfile(src):
        shutil.move(src, dst)
        print(f"Moved {filename}")
