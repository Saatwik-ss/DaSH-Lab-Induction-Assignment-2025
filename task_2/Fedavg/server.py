# server.py
import flwr as fl
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=5)
    args = parser.parse_args()

    # Compatible FedAvg strategy for modern Flower versions
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                  # sample all clients each round
        min_fit_clients=args.num_clients,  # require all clients
        min_available_clients=args.num_clients,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
