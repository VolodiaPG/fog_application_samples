from collections import defaultdict

FOG_NODE_DEPLOYMENT = """apiVersion: v1
kind: ServiceAccount
metadata:
  name: fog-node
  namespace: openfaas
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fog-node
  namespace: openfaas
rules:
  - apiGroups: ["metrics.k8s.io", ""]
    resources: ["pods", "nodes"]
    verbs: ["get", "list", "watch"]
---
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: fog-node
  namespace: openfaas
subjects:
- kind: ServiceAccount
  name: fog-node
  namespace: openfaas
roleRef:
  kind: ClusterRole
  name: fog-node
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: v1
kind: Service
metadata:
  name: fog-node
  namespace: openfaas
  labels:
    app: fog-node
spec:
  type: NodePort
  ports:
    - name: proxied-fog-node-30003
      port: 30003
      targetPort: 30003
      protocol: TCP
      nodePort: 30003
  selector:
    app: fog-node
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fog-node
  namespace: openfaas
  labels:
    app: fog-node
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fog-node
  template:
    metadata:
      labels:
        app: fog-node
    spec:
      shareProcessNamespace: true
      serviceAccountName: fog-node
      automountServiceAccountToken: true
      containers:
      - name: fog-node
        image: {fog_node_image}
        env:
        - name: OPENFAAS_USERNAME
          valueFrom:
            secretKeyRef:
              name: basic-auth
              key: basic-auth-user
        - name: OPENFAAS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: basic-auth
              key: basic-auth-password
        - name: OPENFAAS_IP
          value: "gateway.openfaas"
        - name: OPENFAAS_PORT
          value: "31112"
        - name: CONFIG
          value: "{conf}"
        - name: LOG_CONFIG_PATH
          value: "/var/log"
        - name: LOG_CONFIG_FILENAME
          value: "{node_name}.log"
        - name: RUST_LOG
          value: "warn,fog_node=trace,openfaas=trace,kube_metrics=trace,helper=trace"
        - name: COLLECTOR_IP
          value: "{collector_ip}"
        - name: COLLECTOR_PORT
          value: "14268"
        - name: VALUATION_PER_MILLICPU
          value: "{valuation_per_millicpu}"
        - name: VALUATION_PER_MIB
          value: "{valuation_per_mib}"
        - name: IS_CLOUD
          value: "{is_cloud}"
        ports:
        - containerPort: 30003
        volumeMounts:
        - name: log-storage-fog-node
          mountPath: /var/log
      - name: sidecar-logs
        image: ghcr.io/volodiapg/busybox:latest
        args: [/bin/sh, -c, 'tail -n+1 -F /mnt/log/{node_name}.log']
        volumeMounts:
        - name: log-storage-fog-node
          readOnly: true
          mountPath: /mnt/log
      volumes:
      - name: log-storage-fog-node
        emptyDir: {{}}
"""

MARKET_DEPLOYMENT = """apiVersion: v1
kind: Service
metadata:
  name: market
  namespace: openfaas
  labels:
    app: market
spec:
  type: NodePort
  ports:
    - name: proxied-market-30008
      port: 30008
      targetPort: 30008
      protocol: TCP
      nodePort: 30008
  selector:
    app: market
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market
  namespace: openfaas
  labels:
    app: market
spec:
  replicas: 1
  selector:
    matchLabels:
      app: market
  template:
    metadata:
      labels:
        app: market
    spec:
      containers:
      - name: market
        image: {market_image}
        ports:
        - containerPort: 30008
        - containerPort: 6831
        - containerPort: 6832
        env:
        - name: LOG_CONFIG_PATH
          value: "/var/log"
        - name: LOG_CONFIG_FILENAME
          value: "market.log"
        - name: RUST_LOG
          value: "warn,market=trace"
        - name: SERVER_PORT
          value: "30008"
        - name: COLLECTOR_IP
          value: "{collector_ip}"
        - name: COLLECTOR_PORT
          value: "14268"
        volumeMounts:
        - name: log-storage-market
          mountPath: /var/log
      - name: sidecar-logs
        image: ghcr.io/volodiapg/busybox:latest
        args: [/bin/sh, -c, 'tail -n+1 -F /mnt/log/market.log']
        volumeMounts:
        - name: log-storage-market
          readOnly: true
          mountPath: /mnt/log
      volumes:
      - name: log-storage-market
        emptyDir: {{}}
"""

MARKET_CONNECTED_NODE = """(
    situation: MarketConnected (
      market_ip: "{market_ip}",
      market_port: "30008",
    ),
    my_id: "{my_id}",
    my_public_ip: "{my_public_ip}",
    my_public_port_http: "30003",
    reserved_cpu: "{reserved_cpu} cpus",
    reserved_memory: "{reserved_memory} MiB",
    tags: ["node_to_market", "{name}"],
)

"""

NODE_CONNECTED_NODE = """(
    situation: NodeConnected (
      parent_id: "{parent_id}",
      parent_node_ip: "{parent_ip}",
      parent_node_port_http: "30003",
    ),
    my_id: "{my_id}",
    my_public_ip: "{my_public_ip}",
    my_public_port_http: "30003",
    reserved_cpu: "{reserved_cpu} cpus",
    reserved_memory: "{reserved_memory} MiB",
    tags: ["node_to_node", "{name}"],
)

"""

# Remove a unit so that the hosts are not saturated
NB_CPU_PER_MACHINE_PER_CLUSTER = {
    "gros": {"core": (2 * 18) - 1, "mem": 1024 * (96 - 4)},
    "paravance": {"core": (2 * 8 * 2) - 1, "mem": 1024 * (128 - 4)},
    # "dahu": {"core": 2 * 16 - 1, "mem": 1024 * (192 - 4)},
}

TIER_3_FLAVOR = {
    "core": 2,
    "mem": 1024 * 4,
    "reserved_core": 1.75,
    "reserved_mem": 1024 * 3,
    "valuation_per_mib": 1,
    "valuation_per_millicpu": 1,
    # "valuation_per_mib": 1.5,
    # "valuation_per_millicpu": 1.5,
}
TIER_2_FLAVOR = {
    "core": 6,
    "mem": 1024 * 16,
    "reserved_core": 5,
    "reserved_mem": 1024 * 14,
    "valuation_per_mib": 0.5,
    "valuation_per_millicpu": 0.5,
    # "valuation_per_mib": 1.1,
    # "valuation_per_millicpu": 1.1,
}
TIER_1_FLAVOR = {
    "core": 15,
    "mem": 1024 * 46,
    "reserved_core": 16,
    "reserved_mem": 1024 * 60,
    "valuation_per_mib": 0.1,
    "valuation_per_millicpu": 0.1,
    "is_cloud": True
    # "valuation_per_mib": 0.9,
    # "valuation_per_millicpu": 0.9,
}

NETWORK = {
    "name": "market",
    "flavor": TIER_1_FLAVOR,
    "children": [
        {"name": "marseille", "flavor": TIER_1_FLAVOR, "latency": 6, "children": []},
        {"name": "toulouse", "flavor": TIER_1_FLAVOR, "latency": 4, "children": []},
        {
            "name": "paris",
            "flavor": TIER_1_FLAVOR,
            "latency": 3,
            "children": [
                {
                    "name": "rennes",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 10,
                    "children": [
                        {
                            "name": "st-greg",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "st-greg-1-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "st-greg-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 7,
                                },
                                {
                                    "name": "st-greg-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "st-greg-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "st-greg-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,  # ms
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                        {
                            "name": "cesson",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "cesson-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "cesson-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "cesson-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "cesson-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "cesson-3-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "nantes",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 17,
                    "children": [
                        {
                            "name": "orvault",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 10,
                            "children": [
                                {
                                    "name": "orvault-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "orvault-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,
                                },
                                {
                                    "name": "orvault-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                            ],
                        },
                        {
                            "name": "vertou",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 15,
                            "children": [
                                {
                                    "name": "vertou-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "vertou-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "vertou-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "limoux",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 25,
                    "children": [
                        {
                            "name": "roquefeuil",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 31,
                            "children": [
                                {
                                    "name": "roquefeuil-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 1,
                                },
                            ],
                        },
                        {
                            "name": "belcaire",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 29,
                            "children": [
                                {
                                    "name": "belcaire-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "belcaire-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                            ],
                        },
                        {
                            "name": "espezel",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 36,
                            "children": [
                                {
                                    "name": "espezel-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "espezel-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "espezel-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "espezel-4",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "espezel-5",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "espezel-6-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "name": "edinburgh",
            "flavor": TIER_1_FLAVOR,
            "latency": 3,
            "children": [
                {
                    "name": "aberdeen",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 10,
                    "children": [
                        {
                            "name": "dunfermline",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "dunfermline-1-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "dunfermline-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 7,
                                },
                                {
                                    "name": "dunfermline-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "dunfermline-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "dunfermline-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,  # ms
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                        {
                            "name": "glasgow",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "glasgow-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "glasgow-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "glasgow-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "glasgow-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "glasgow-3-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "Inverness",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 17,
                    "children": [
                        {
                            "name": "stirling",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 10,
                            "children": [
                                {
                                    "name": "stirling-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "stirling-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,
                                },
                                {
                                    "name": "stirling-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                            ],
                        },
                        {
                            "name": "st-andrews",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 15,
                            "children": [
                                {
                                    "name": "st-andrews-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "st-andrews-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "st-andrews-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "perth",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 25,
                    "children": [
                        {
                            "name": "perth-lower",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 31,
                            "children": [
                                {
                                    "name": "perth-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 1,
                                },
                            ],
                        },
                        {
                            "name": "ayr",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 29,
                            "children": [
                                {
                                    "name": "ayr-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "ayr-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                            ],
                        },
                        {
                            "name": "dumfries",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 36,
                            "children": [
                                {
                                    "name": "dumfries-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "dumfries-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "dumfries-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "dumfries-4",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "dumfries-5",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "dumfries-6-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "name": "rome",
            "flavor": TIER_1_FLAVOR,
            "latency": 3,
            "children": [
                {
                    "name": "milan",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 10,
                    "children": [
                        {
                            "name": "naples",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "naples-1-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "naples-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 7,
                                },
                                {
                                    "name": "naples-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "naples-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "naples-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,  # ms
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                        {
                            "name": "turin",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "turin-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "turin-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "turin-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "turin-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "turin-3-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "palermo",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 17,
                    "children": [
                        {
                            "name": "genoa",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 10,
                            "children": [
                                {
                                    "name": "genoa-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "genoa-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,
                                },
                                {
                                    "name": "genoa-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                            ],
                        },
                        {
                            "name": "bologna",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 15,
                            "children": [
                                {
                                    "name": "bologna-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "bologna-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "bologna-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "florence",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 25,
                    "children": [
                        {
                            "name": "florence-lower",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 31,
                            "children": [
                                {
                                    "name": "florence-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 1,
                                },
                            ],
                        },
                        {
                            "name": "bari",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 29,
                            "children": [
                                {
                                    "name": "bari-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "bari-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                            ],
                        },
                        {
                            "name": "catania",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 36,
                            "children": [
                                {
                                    "name": "catania-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "catania-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "catania-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "catania-4",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "catania-5",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "catania-6-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "name": "barcelona",
            "flavor": TIER_1_FLAVOR,
            "latency": 3,
            "children": [
                {
                    "name": "granada",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 10,
                    "children": [
                        {
                            "name": "madrid",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "madrid-1-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "madrid-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 7,
                                },
                                {
                                    "name": "madrid-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "madrid-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "madrid-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,  # ms
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                        {
                            "name": "valencia",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,
                            "children": [
                                {
                                    "name": "valencia-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "valencia-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "valencia-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                                {
                                    "name": "valencia-2-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "valencia-3-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 2,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "bilbao",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 17,
                    "children": [
                        {
                            "name": "cordoba",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 10,
                            "children": [
                                {
                                    "name": "cordoba-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "cordoba-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,
                                },
                                {
                                    "name": "cordoba-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                            ],
                        },
                        {
                            "name": "seville",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 15,
                            "children": [
                                {
                                    "name": "seville-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 10,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "seville-1",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,
                                },
                                {
                                    "name": "seville-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 5,
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "malaga",
                    "flavor": TIER_2_FLAVOR,
                    "latency": 25,
                    "children": [
                        {
                            "name": "cadiz",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 31,
                            "children": [
                                {
                                    "name": "cadiz-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 1,
                                },
                            ],
                        },
                        {
                            "name": "toledo",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 29,
                            "children": [
                                {
                                    "name": "toledo-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "toledo-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                            ],
                        },
                        {
                            "name": "zaragoza",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 36,
                            "children": [
                                {
                                    "name": "zaragoza-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 3,  # ms
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "zaragoza-2",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "zaragoza-3",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                                {
                                    "name": "zaragoza-4",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "zaragoza-5",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                },
                                {
                                    "name": "zaragoza-6-in",
                                    "flavor": TIER_3_FLAVOR,
                                    "latency": 4,
                                    "iot_connected": 0,  # ms
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ],
}

NETWORK = {
    "name": "market",
    "flavor": TIER_1_FLAVOR,
    "children": [
        {
            "name": "rennes",
            "flavor": TIER_2_FLAVOR,
            "latency": 10,
            "children": [
                {
                    "name": "st-greg",
                    "flavor": TIER_3_FLAVOR,
                    "latency": 17,
                    "children": [
                        {
                            "name": "st-greg-1-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 3,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "st-greg-2-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 1,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "st-greg-3-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "st-greg-4-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 5,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "st-greg-5",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 5,  # ms
                        },
                        {
                            "name": "cesson-6-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 3,  # ms
                            "iot_connected": 0,  # ms
                        },
                    ],
                },
                {
                    "name": "cesson",
                    "flavor": TIER_3_FLAVOR,
                    "latency": 7,
                    "children": [
                        {
                            "name": "cesson-1-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 3,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-2-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 1,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-3-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 7,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-4-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 5,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-6-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 6,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-7-in",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 4,  # ms
                            "iot_connected": 0,  # ms
                        },
                        {
                            "name": "cesson-5",
                            "flavor": TIER_3_FLAVOR,
                            "latency": 3,  # ms
                        },
                    ],
                },
            ],
        },
    ],
}


def flatten(container):
    for i in container:
        if isinstance(i, list):
            for j in flatten(i):
                yield j
        else:
            yield i


def gen_fog_nodes_names(node):
    name = node["name"]

    children = node["children"] if "children" in node else []

    return [name, *[gen_fog_nodes_names(node) for node in children]]


def get_extremities_name(node):
    name = node["name"]

    children = node["children"] if "children" in node else []

    ret = [get_extremities_name(node) for node in children]
    if len(children) == 0:
        ret.append(name)

    return ret


def get_iot_connection(node):
    name = node["name"]

    children = node["children"] if "children" in node else []

    ret = [get_iot_connection(node) for node in children]
    if "iot_connected" in node:
        ret.append((name, node["iot_connected"]))
    return ret


def adjacency(node):
    children = node["children"] if "children" in node else []
    ret = {}
    ret[node["name"]] = [(child["name"], child["latency"]) for child in children]
    for child in children:
        ret = {**ret, **adjacency(child)}

    return ret


def levels(node, level=0):
    children = node["children"] if "children" in node else []
    ret = {}
    ret[node["name"]] = level
    for child in children:
        ret = {**ret, **levels(child, level + 1)}

    return ret


def adjacency_undirected(node):
    ret = defaultdict(lambda: [])

    def fun(node):
        children = node["children"] if "children" in node else []
        for child in children:
            ret[node["name"]] += [(child["name"], child["latency"])]
            ret[child["name"]] += [(node["name"], child["latency"])]
            fun(child)

    fun(node)
    return ret


FOG_NODES = list(flatten([gen_fog_nodes_names(child) for child in NETWORK["children"]]))
EXTREMITIES = list(
    flatten([get_extremities_name(child) for child in NETWORK["children"]])
)
IOT_CONNECTION = list(
    flatten([get_iot_connection(child) for child in NETWORK["children"]])
)
ADJACENCY = adjacency(NETWORK)
LEVELS = levels(NETWORK)
