use std::os::unix::process::CommandExt;

fn main() {
    let error = match std::env::current_exe() {
        Ok(exe) => std::process::Command::new(exe.with_file_name("flyingfish"))
            .arg0("ff")
            .args(std::env::args_os().skip(1))
            .exec(),
        Err(error) => error,
    };
    eprintln!("error: ff runs the flyingfish binary in its own directory: {error}");
    std::process::exit(1);
}
