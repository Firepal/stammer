import subprocess
import io

class VideoOut:
    def __init__(self, temp_dir: str, framerate: int, output_path: str):
        self.temp_dir = temp_dir
        self.framerate = framerate
        self.output_path = output_path
        self.frames_written = 0

        self.proc = None

    def _get_output_cmd(self,input = None):
        if input == None:
            input = [
                '-f', 'image2pipe', '-i', 'pipe:',
                '-i', str(self.temp_dir / 'out.wav')
            ]
        cmd = [
            'ffmpeg',
            '-v', 'quiet',
            '-y',
            '-framerate', str(self.framerate),
            '!inputs!',
            '-c:a', 'aac',
            '-c:v', 'libx264',
            '-crf', '24',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            str(self.output_path)
        ]

        def replace(value, list):
            idx = cmd.index(value)
            cmd.pop(idx)
            for i, x in enumerate(list): cmd.insert(idx+i,x)

        replace('!inputs!',input)
    
        return cmd
    
    def _create_output_proc(self):
        call = self._get_output_cmd()

        return subprocess.Popen(
            call,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL
        )

    # def get_progress_strings(self) -> list[str]:
    #     strings: list[str] = []
    #     strings.append(str(self.frames_written) + "/" + str(self.best_match_count))
        
    #     return strings
    
    # def progress_strings_separated(self):
    #     ps = self.get_progress_strings()
    #     if len(ps) == 1: return ps[0]
    #     return " . ".join(self.get_progress_strings())
    
    # def print_progress(self):
    #     print(self.progress_strings_separated(),end='      \r')
    
    # def notify_write(self):
    #     self.frames_written += 1
    #     self.print_progress()
    
    def write_frame(self,frame: io.BytesIO):
        if self.proc == None:
            self.proc = self._create_output_proc()

        frame.seek(0)
        f = frame.read()
        self.proc.stdin.write(f)
        # self.notify_write()
    
    def complete(self):
        self.proc.communicate()