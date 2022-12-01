import os

from azure.storage.filedatalake import DataLakeServiceClient


# manage adding data to the data lake

class DataLake:
    service_client: DataLakeServiceClient

    def __init__(self, account_name: str, account_key: str):
        try:
            url = "https://{}.dfs.core.windows.net".format(account_name)
            self.service_client = DataLakeServiceClient(account_url=url, credential=account_key)
        except Exception as e:
            print(e)

    def list_directory_contents(self, file_system_name: str, directory_name: str):
        try:
            print("List directory contents")
            file_system_client = self.service_client.get_file_system_client(file_system=file_system_name)

            paths = file_system_client.get_paths(path=directory_name)

            print("Paths:")

            for path in paths:
                print('hello')
                print(path.name + '\n')

        except Exception as e:
            print(e)

    def upload_file(self, file_system_name: str, directory_name: str, file_name: str, local_file_path: str):
        print("Upload file to directory", directory_name, "in file system", file_system_name, "as", file_name)
        try:
            file_system_client = self.service_client.get_file_system_client(file_system=file_system_name)
            directory_client = file_system_client.get_directory_client(directory_name)
            file_client = directory_client.create_file(file_name)
            file_client.append_data(data=open(local_file_path, 'rb'), offset=0, length=os.path.getsize(local_file_path))
            file_client.flush_data(len(open(local_file_path, 'rb').read()))
        except Exception as e:
            print(e)


