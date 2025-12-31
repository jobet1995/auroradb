use auroradb::core::variables;

fn main() {
    println!("{} v{}", variables::DATABASE_NAME, variables::DATABASE_VERSION);
}
